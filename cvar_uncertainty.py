"""The paper's CVaR case study (`sim.py`) with the model's own uncertainty added to the energy distribution.

    python uncertainty.py          # first: writes the recalibration factors to Results/uncertainty/calibration.csv
    python cvar_uncertainty.py     # -> Results/uncertainty/cvar.csv, cvar.png

As in `sim.py`: --n_sims Monte-Carlo flights of the case-study path (QuadSim, random inlet wind), the energy of
each predicted by the model, and the risk from `Risk.Risk` with the paper's parameters (99 %, a=64, b=92.34,
B=369.36 kJ). The paper's version only has the wind spread: the model's prediction counts as exact.
Here every simulated flight's energy becomes a Gaussian from the method's (recalibrated) predictive
distribution, and --draws samples of it per flight go into the risk instead.

The case-study path takes ~1 min and ~28 kJ, 8 % of B, where the risk profile is flat, so the uncertainty barely
moves its CVaR. To see where it matters, the energies are also scaled to k laps of the path (k = 1 .. --max_laps,
same wind and same model error on every lap): cvar_laps.csv / the right panel of cvar.png give CVaR against
mission length, and `max_laps` is the longest mission whose 99th-percentile energy still leaves the reserve b.

--split <held-out split> (e.g. wind): the runs of that split and of its control (e.g. random_sub_wind), with the
recalibration factors of `uncertainty.py --split wind`. A simulation has no true energy, so the control (which saw
that condition in training) is the reference for how far off the held-out models' risk is. --inlet_mag sets the
simulated wind (the paper: 3 +- 1.5 m/s at the inlet), e.g. a windy day:

    python uncertainty.py --split wind
    python cvar_uncertainty.py --split wind --inlet_mag 6 1

Differences from sim.py: the models are the PyTorch checkpoints, and the inputs are normalized with the stats
stored in the checkpoint (sim.py renormalized the inputs with the simulated flights' own range).
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from QuadSim import QuadSim
from Risk import Risk
from data_utils import process_data, sim_to_network_transform
from torch_data import create_tensors, predict_flight
from torch_model import load_model
from uncertainty import FIRST, control_of, flight_energy, method_paths

# the case study of sim.py: waypoints (x, y, z, time), 10 m/s, inlet wind N(-2.53 deg, 28.47) and N(3, 1.5) m/s
WAYPOINTS = np.array([[-259.9, -260., -260., -200., -65., -64.9],
                      [125., 125., 50., -35., 100., 100.],
                      [0., 20., 20., 20., 20., 0.],
                      [0., 3., 3., 3., 3., 5.]])
AIRSPEED = 10
INLET_ANG = (-2.53455, 28.4662)
INLET_MAG = (3.0, 1.5)
RISK = {'limit': 99., 'a': 64, 'b': 92.34, 'B': 369.36}


def simulate(n, seed, inlet_mag=INLET_MAG, keep_failed=False):
    """`sim.mc_flights` without the process pool: the states of n simulated flights and how many failed.

    A failed flight (the wind is too strong to reach the goal) keeps flying until the simulator gives up, so its
    energy says nothing about the model; it is dropped unless keep_failed (sim.py keeps it).
    """
    wind_ang = np.load('wind_data/dataset_hawkins.npy')
    wind_mag = np.load('wind_data/dataset_hawkins_mag.npy')
    rng = np.random.default_rng(seed)
    np.random.seed(seed)  # QuadSim's turbulence uses the global generator
    flights, fails = [], 0
    for _ in range(n):
        sim = QuadSim(init_pos=WAYPOINTS[:, 0], wind_map_ang=wind_ang, wind_map_mag=wind_mag,
                      command_airspeed=True, ghost_delta=1.0)
        sim.init_path_from_waypoints(WAYPOINTS, airspeed=AIRSPEED)
        sim.update_inlet(rng.normal(*INLET_ANG), rng.normal(*inlet_mag), 'rbf')
        states, _, _, success = sim.propagate()
        fails += not success
        if success or keep_failed:
            flights.append(sim_to_network_transform(states))
    return flights, fails


def sim_member_energies(paths, flights, device, mc_samples=0):
    """(K, n) member energies and (K, n) member noise stds in kJ for the simulated flights."""
    sims = dict(enumerate(flights))
    energies, sds = [], []
    for path in paths:
        model, inputs, ckpt = load_model(path, map_location=device)
        data, data_min, data_max, _, _, _ = process_data(sims, eval_mode=True,
                                                         stats=(ckpt['data_min'], ckpt['data_max']), **inputs)
        tensors = create_tensors(data, list(sims), device=device, **inputs)
        scale, offset = data_max[-1] - data_min[-1], data_min[-1]
        for _ in range(max(mc_samples, 1)):
            e, sd = [], []
            for i, states in sims.items():
                mu, var = predict_flight(model, *tensors[i][:2], inputs['batch_size'], inputs['stateful'],
                                         return_var=True, mc_dropout=mc_samples > 0)
                ei, sdi = flight_energy((mu * scale + offset)[None], (var * scale ** 2)[None],
                                        states['time'].values[FIRST:])
                e.append(ei[0])
                sd.append(sdi[0])
            energies.append(e)
            sds.append(sd)
    return np.array(energies), np.array(sds)


def risk_row(name, energies):
    r = Risk(energies, **RISK)
    return {'method': name, 'n_energies': len(energies), 'mean_kJ': energies.mean(), 'p99_kJ': np.percentile(energies, 99),
            'var99': r.var(), 'cvar99': r.cvar(), 'p_over_B': np.mean(energies > RISK['B']) * 100}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', default='tcn_20_64_2_1_5', help='Run name of the seed-0 plain model (random split).')
    parser.add_argument('--split', default='random', help='random, or a held-out split such as wind (see the docstring).')
    parser.add_argument('--ckpt_dir', default='Results/ckpts')
    parser.add_argument('--calibration', default=None,
                        help='From uncertainty.py (default Results/uncertainty[_<split>]/calibration.csv); gives each '
                             'method\'s energy recalibration factor.')
    parser.add_argument('--inlet_mag', default=list(INLET_MAG), nargs=2, type=float, metavar=('MEAN', 'STD'),
                        help='Inlet wind speed of the simulations in m/s (default: the paper\'s 3 1.5).')
    parser.add_argument('--sigma', default='recalibrated', choices=['raw', 'recalibrated'],
                        help='Energy std as predicted (raw: noise taken as fully correlated over the flight, a '
                             'conservative bound) or scaled to cover 90 %% of the validation flights.')
    parser.add_argument('--n_sims', default=200, type=int, help='Monte-Carlo flights (sim.py uses 50).')
    parser.add_argument('--draws', default=100, type=int, help='Energy samples per flight from the predictive distribution.')
    parser.add_argument('--mc_samples', default=30, type=int, help='Forward passes for MC dropout.')
    parser.add_argument('--keep_failed', action='store_true',
                        help='Keep the simulated flights that did not reach the goal (sim.py does); they are dropped by default.')
    parser.add_argument('--max_laps', default=12, type=int, help='Longest mission for the lap sweep, in laps of the path.')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--out', default=None, help='Output folder (default Results/uncertainty[_<split>]).')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    folder = 'Results/uncertainty' + ('' if args.split == 'random' else '_' + args.split)
    out = args.out or folder
    calib = pd.read_csv(args.calibration or os.path.join(folder, 'calibration.csv'))
    calib = calib[calib['sigma'] == args.sigma]
    if 'split' not in calib:  # random-split calibration from before --split
        calib = calib.assign(split='random')
    calib = calib.set_index(['split', 'method'])['energy_scale']

    # (label, split, method, checkpoints, MC samples) for every method with checkpoints and a recalibration factor
    splits = ['random'] if args.split == 'random' else [args.split, control_of(args.split)]
    runs = [(m if args.split == 'random' else '{} ({})'.format(m, s), s, m, paths, mc)
            for s in splits for m, (paths, mc) in method_paths(args.prefix, args.ckpt_dir, s).items()
            if (s, m) in calib.index]
    if not runs:
        raise SystemExit('no method has both checkpoints and a row in the calibration file; run uncertainty.py first')

    flights, fails = simulate(args.n_sims, args.seed, args.inlet_mag, args.keep_failed)
    print('{} simulated flights, inlet wind {} +- {} m/s: {} did not reach the goal{}'.format(
        args.n_sims, *args.inlet_mag, fails, '' if args.keep_failed else ' (dropped)'))

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)  # MC dropout masks
    rows, samples = [], {}
    for name, split, method, paths, mc in runs:
        mc = args.mc_samples if mc is None else mc
        e, sd = sim_member_energies(paths, flights, args.device, mc)
        if method == 'ensemble':  # the paper's estimate: one model, its prediction taken as exact
            point = 'point (seed 0, paper)' if args.split == 'random' else 'point ({})'.format(split)
            rows.append(risk_row(point, e[0]))
            samples[point] = e[0]
        # per flight: moment-matched Gaussian of the members, its std scaled by the validation factor
        mean = e.mean(0)
        std = np.sqrt((sd ** 2).mean(0) + e.var(0)) * calib[split, method]
        draws = (mean[:, None] + std[:, None] * rng.standard_normal((len(mean), args.draws))).ravel()
        rows.append(risk_row(name, draws))
        rows[-1]['mean_sd_kJ'] = std.mean()
        samples[name] = draws

    # lap sweep: k laps -> k times the energy of each sample
    laps = np.arange(1, args.max_laps + 1)
    reserve_limit = RISK['B'] - RISK['b']  # energy above which the reserve b is used
    sweep = pd.DataFrame([{'method': name, 'laps': k, 'mean_kJ': k * s.mean(), 'p99_kJ': k * np.percentile(s, 99),
                           'cvar99': Risk(k * s, **RISK).cvar()} for name, s in samples.items() for k in laps])
    for row in rows:
        ok = sweep[(sweep['method'] == row['method']) & (sweep['p99_kJ'] <= reserve_limit)]
        row['max_laps'] = int(ok['laps'].max()) if len(ok) else 0

    table = pd.DataFrame(rows)
    tag = ('' if args.sigma == 'recalibrated' else '_' + args.sigma) + \
        ('' if tuple(args.inlet_mag) == INLET_MAG else '_inlet{:g}'.format(args.inlet_mag[0]))
    args.out = out
    os.makedirs(args.out, exist_ok=True)
    table.to_csv(os.path.join(args.out, 'cvar{}.csv'.format(tag)), index=False)
    sweep.to_csv(os.path.join(args.out, 'cvar_laps{}.csv'.format(tag)), index=False)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = np.linspace(min(s.min() for s in samples.values()), max(s.max() for s in samples.values()), 60)
    for name, s in samples.items():
        ax.hist(s, bins, density=True, histtype='step', lw=1.5, label=name)
        d = sweep[sweep['method'] == name]
        ax2.plot(d['mean_kJ'], d['cvar99'], marker='o', ms=3, label=name)
    ax.set(xlabel='predicted flight energy (kJ)', ylabel='density', title='Case-study energy, {} MC flights'.format(len(flights)))
    ax2.set(xlabel='mean mission energy (kJ), 1..{} laps'.format(args.max_laps), ylabel='CVaR 99 % (risk)',
            title='Risk vs mission length (B = {} kJ)'.format(RISK['B']))
    for a in (ax, ax2):
        a.legend()
        a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'cvar{}.png'.format(tag)), dpi=150)

    pd.set_option('display.width', 200)
    print(table.round(3).to_string(index=False))
    print('\nCVaR 99 % by mission length (laps of the path):')
    print(sweep.pivot(index='laps', columns='method', values='cvar99').round(3).to_string())
    print('saved', args.out)
