"""Predictive uncertainty of the deep energy model: deep ensemble, variance head, MC dropout.

    python uncertainty.py                                   # b-TCN runs in Results/ckpts -> Results/uncertainty/
    python uncertainty.py --prefix tcn_20_16_2_1_5          # s-TCN runs instead

Every method turns checkpoints into a Gaussian predictive distribution N(mu, sigma^2) of the power at each step:

    ensemble       <prefix>, <prefix>_s1, <prefix>_s2 (plain MSE models): mu = mean of the members, sigma = their
                   spread. Only the model's own uncertainty (epistemic); measurement noise is not in it.
    var_head       <prefix>_var (seed 0), trained with the Gaussian NLL: sigma = the predicted standard deviation.
                   Only the noise (aleatoric).
    var_ensemble   the 3 variance-head seeds as one mixture: sigma^2 = mean member variance + spread of the means.
    mc_dropout     <prefix>_d0.1 (seed 0), --mc_samples passes with dropout on: sigma = spread of the passes.

Each method is also recalibrated: sigma is scaled by the factor that makes the 90 % interval cover 90 % of its own
validation steps (inside the training conditions, so a held-out condition is never used for calibration).

--split random (default): runs trained on the paper's split, scored on its test flights.
--split <held-out split>, e.g. wind: the runs of that split (<prefix>_wind, <prefix>_var_wind, ...) and of its control
(e.g. random_sub_wind), all scored on the same flights: the split's test flights that no run of either trained or
validated on (the flights stress_test.py scores). Default --out: Results/uncertainty_<split>.

    python uncertainty.py --split wind

Outputs in --out:
    calibration.csv     per split and method (raw / recalibrated): MAPE, coverage of the 50/80/90/95 % intervals,
                        mean 90 % width, NLL, the flight-energy coverage, and the energy bias (predicted - true, %)
    per_flight.csv      per method and test flight: wind, RMSE, mean sigma, 90 % coverage, energy z-score
    wind.csv            per method: Spearman correlation of wind with mean sigma and with RMSE, calm vs windy
    calibration.png     coverage vs nominal level;   wind.png  mean sigma and RMSE vs wind per flight
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm, spearmanr

from data_utils import get_flights_and_ranges, load_dataset, process_data
from splits import CONTROLS, WIND_THRESHOLD, get_split, load_conditions
from torch_data import create_tensors, predict_flight
from torch_model import load_model

LEVELS = [.5, .8, .9, .95]
FIRST = 19  # lookback - 1: the first predicted step


def member_paths(prefix, seeds, ckpt_dir):
    return [os.path.join(ckpt_dir, prefix + ('' if s == 0 else '_s{}'.format(s)), 'final.pt') for s in seeds]


def method_paths(prefix, ckpt_dir, split='random'):
    """{method: (checkpoints, MC samples)} for the methods whose checkpoints exist (run names as in train_torch.py)."""
    s = '' if split == 'random' else '_' + split
    methods = {
        'ensemble': (member_paths(prefix + s, [0, 1, 2], ckpt_dir), 0),
        'var_head': (member_paths(prefix + '_var' + s, [0], ckpt_dir), 0),
        'var_ensemble': (member_paths(prefix + '_var' + s, [0, 1, 2], ckpt_dir), 0),
        'mc_dropout': (member_paths(prefix + '_d0.1' + s, [0], ckpt_dir), None),
    }
    return {m: v for m, v in methods.items() if all(os.path.exists(p) for p in v[0])}


def control_of(split):
    """The data-size control a held-out split is compared with (as in stress_test.py)."""
    return next((c for c, s in CONTROLS.items() if s == split), 'random_subsampled')


def seen_flights(paths):
    """Flights any of the checkpoints trained or validated on."""
    seen = set()
    for p in paths:
        ranges = torch.load(p, map_location='cpu', weights_only=True).get('ranges')
        if ranges is not None:
            seen |= set(ranges['train'] + ranges['val'])
    return seen


def member_predictions(paths, all_data, flights, device, mc_samples=0):
    """{flight: (means (K, N), variances (K, N))} in W and W^2, one row per member (or MC pass)."""
    out = {f: ([], []) for f in flights}
    for path in paths:
        model, inputs, ckpt = load_model(path, map_location=device)
        stats = (ckpt['data_min'], ckpt['data_max']) if 'data_min' in ckpt else None
        data, data_min, data_max, _, _, _ = process_data(all_data, stats=stats, **inputs)
        tensors = create_tensors(data, flights, device=device, **inputs)
        scale, offset = data_max[-1] - data_min[-1], data_min[-1]
        for f in flights:
            for _ in range(max(mc_samples, 1)):
                mu, var = predict_flight(model, *tensors[f][:2], inputs['batch_size'], inputs['stateful'],
                                         return_var=True, mc_dropout=mc_samples > 0)
                out[f][0].append(mu * scale + offset)
                out[f][1].append(var * scale ** 2)
    return {f: (np.array(m), np.array(v)) for f, (m, v) in out.items()}


def gaussian(means, variances):
    """Moment-matched Gaussian of an equally weighted mixture: (mu, sigma) per step."""
    mu = means.mean(0)
    return mu, np.sqrt(variances.mean(0) + means.var(0))


def flight_energy(means, variances, ts):
    """Energy (kJ) of each member and its noise std (kJ), treating a member's noise as fully correlated in time.

    This is the conservative bound. Measured on the b-TCN test residuals, the per-step correlation is short (0.48 at
    lag 1, 0.15 at lag 10 steps), but each flight also has a persistent offset, so the flight-energy error is ~7x
    what independent per-step noise would give: the truth lies between the two bounds. The energy recalibration
    in `__main__` (on the validation flights) corrects for that.
    """
    e = np.trapezoid(means, ts, axis=1) / 1000
    e_sd = np.trapezoid(np.sqrt(variances), ts, axis=1) / 1000
    return e, e_sd


def energy_gaussian(means, variances, ts):
    """(mean, std) of the predicted flight energy in kJ."""
    e, e_sd = flight_energy(means, variances, ts)
    return e.mean(), np.sqrt((e_sd ** 2).mean() + e.var())


def scale_factor(y, mu, sd, level=.9):
    """Factor for sigma that makes the central `level` interval cover `level` of the given steps."""
    z = np.abs(y - mu) / np.maximum(sd, 1e-9)
    return np.quantile(z, level) / norm.ppf(.5 + level / 2)


def metrics(y, mu, sd, e_true, e_mu, e_sd):
    """Step-level calibration metrics pooled over flights, plus energy-level coverage over flights."""
    y, mu, sd = (np.concatenate(a) for a in (y, mu, sd))
    sd = np.maximum(sd, 1e-9)
    row = {}
    for lvl in LEVELS:
        row['cov{:d}'.format(int(lvl * 100))] = np.mean(np.abs(y - mu) <= norm.ppf(.5 + lvl / 2) * sd) * 100
    row['width90'] = np.mean(2 * norm.ppf(.95) * sd)
    # MC dropout gives sigma ~ 0 at a few steps, which alone would make the NLL astronomical: floor sigma at 1 W
    sd_nll = np.maximum(sd, 1.)
    row['nll'] = np.mean(.5 * np.log(2 * np.pi * sd_nll ** 2) + (y - mu) ** 2 / (2 * sd_nll ** 2))
    z_e = (np.array(e_true) - np.array(e_mu)) / np.maximum(np.array(e_sd), 1e-9)
    row['energy_cov90'] = np.mean(np.abs(z_e) <= norm.ppf(.95)) * 100
    # negative = the model under-predicts the energy, so a risk estimate built on it is optimistic
    row['energy_bias'] = np.mean((np.array(e_mu) - np.array(e_true)) / np.array(e_true)) * 100
    row['energy_sd_kJ'] = np.mean(e_sd)
    return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', default='./', help='Working directory for data.')
    parser.add_argument('--prefix', default='tcn_20_64_2_1_5', help='Run name of the seed-0 plain model (random split).')
    parser.add_argument('--split', default='random', help='random, or a held-out split such as wind (see the docstring).')
    parser.add_argument('--ckpt_dir', default='Results/ckpts')
    parser.add_argument('--mc_samples', default=30, type=int, help='Forward passes for MC dropout.')
    parser.add_argument('--out', default=None, help='Output folder (default Results/uncertainty[_<split>]).')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', default=0, type=int, help='Seed for the MC dropout masks.')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    out = args.out or 'Results/uncertainty' + ('' if args.split == 'random' else '_' + args.split)

    data_dir = os.path.join(args.directory, 'data/')
    all_data = load_dataset(data_dir)
    cond = load_conditions(data_dir)
    wind = cond['wind']

    # (split, methods, validation flights) for every group of runs, and the flights all of them are scored on
    if args.split == 'random':
        _, _, _, test_range, val_range, _ = get_flights_and_ranges(all_data)
        groups = [('random', method_paths(args.prefix, args.ckpt_dir), sorted(set(int(f) for f in val_range)))]
        test = sorted(set(int(f) for f in test_range))
    else:
        groups = []
        for split in [args.split, control_of(args.split)]:
            _, val_range, _ = get_split(all_data, split, data_dir, cond=cond)
            groups.append((split, method_paths(args.prefix, args.ckpt_dir, split), sorted(set(int(f) for f in val_range))))
        test_range = get_split(all_data, args.split, data_dir, cond=cond)[0]
        test = sorted(set(int(f) for f in test_range))
    groups = [g for g in groups if g[1]]
    if not groups:
        raise SystemExit('no checkpoints found for prefix {!r}, split {!r} in {}'.format(args.prefix, args.split, args.ckpt_dir))
    seen = seen_flights([p for _, methods, _ in groups for paths, _ in methods.values() for p in paths])
    if args.split != 'random':
        test = [f for f in test if f not in seen]
    elif seen & set(test):  # every method must be scored on flights none of its members trained or validated on
        raise SystemExit('a random-split run trained on its test flights; check --prefix / --ckpt_dir')
    print('scoring on {} flights: {}'.format(len(test), test))

    vals = sorted({f for _, _, v in groups for f in v})
    truth = {f: (all_data[f]['power'].values[FIRST:], all_data[f]['time'].values[FIRST:]) for f in test + vals}
    os.makedirs(out, exist_ok=True)
    rows, flight_rows, wind_rows, curves = [], [], [], {}
    for split, methods, val in groups:
        for name, (paths, mc) in methods.items():
            label = name if args.split == 'random' else '{} ({})'.format(name, split)
            mc = args.mc_samples if mc is None else mc
            print('{}: {} checkpoint(s){}'.format(label, len(paths), ', {} MC passes'.format(mc) if mc else ''))
            preds = member_predictions(paths, all_data, sorted(set(test + val)), args.device, mc)
            dist = {f: gaussian(*preds[f]) for f in preds}
            energy = {f: energy_gaussian(*preds[f], truth[f][1]) for f in preds}
            e_true = {f: np.trapezoid(*truth[f]) / 1000 for f in preds}

            # recalibrate on the validation flights: one factor for the steps, one for the flight energy
            s = scale_factor(np.concatenate([truth[f][0] for f in val]), np.concatenate([dist[f][0] for f in val]),
                             np.concatenate([dist[f][1] for f in val]))
            z_val = [(e_true[f] - energy[f][0]) / energy[f][1] for f in val]
            s_e = np.quantile(np.abs(z_val), .9) / norm.ppf(.95)

            mape = np.mean([np.mean(np.abs((truth[f][0] - dist[f][0]) / truth[f][0])) * 100 for f in test])
            for sigma, k, k_e in [('raw', 1., 1.), ('recalibrated', s, s_e)]:
                row = {'split': split, 'method': name, 'sigma': sigma, 'scale': k, 'energy_scale': k_e, 'mape': mape}
                row.update(metrics([truth[f][0] for f in test], [dist[f][0] for f in test],
                                   [dist[f][1] * k for f in test], [e_true[f] for f in test],
                                   [energy[f][0] for f in test], [energy[f][1] * k_e for f in test]))
                rows.append(row)
                curves[label, sigma] = np.concatenate([np.abs(truth[f][0] - dist[f][0]) / np.maximum(dist[f][1] * k, 1e-9)
                                                       for f in test])

            per_flight = pd.DataFrame([{
                'split': split, 'method': name, 'label': label, 'flight': f, 'wind': wind.get(f, np.nan),
                'rmse': np.sqrt(np.mean((truth[f][0] - dist[f][0]) ** 2)),
                'mean_sigma': np.mean(dist[f][1]),
                'cov90_recal': np.mean(np.abs(truth[f][0] - dist[f][0]) <= norm.ppf(.95) * s * dist[f][1]) * 100,
                'energy_true_kJ': e_true[f], 'energy_pred_kJ': energy[f][0], 'energy_sd_kJ': energy[f][1],
                'energy_z': (e_true[f] - energy[f][0]) / energy[f][1],
            } for f in test])
            flight_rows.append(per_flight)

            pf = per_flight.dropna(subset=['wind'])
            windy = pf['wind'] > WIND_THRESHOLD
            wind_rows.append({
                'split': split, 'method': name, 'n_calm': int((~windy).sum()), 'n_windy': int(windy.sum()),
                'spearman_wind_sigma': spearmanr(pf['wind'], pf['mean_sigma'])[0] if len(pf) > 2 else np.nan,
                'spearman_wind_rmse': spearmanr(pf['wind'], pf['rmse'])[0] if len(pf) > 2 else np.nan,
                'sigma_calm': pf.loc[~windy, 'mean_sigma'].mean(), 'sigma_windy': pf.loc[windy, 'mean_sigma'].mean(),
                'rmse_calm': pf.loc[~windy, 'rmse'].mean(), 'rmse_windy': pf.loc[windy, 'rmse'].mean(),
                'cov90_recal_calm': pf.loc[~windy, 'cov90_recal'].mean(),
                'cov90_recal_windy': pf.loc[windy, 'cov90_recal'].mean(),
            })

    calib = pd.DataFrame(rows)
    per_flight = pd.concat(flight_rows)
    wind_df = pd.DataFrame(wind_rows)
    if args.split == 'random':  # keep the original columns for the random split
        calib, wind_df = calib.drop(columns='split'), wind_df.drop(columns='split')
        per_flight = per_flight.drop(columns=['split', 'label'])
    calib.to_csv(os.path.join(out, 'calibration.csv'), index=False)
    per_flight.to_csv(os.path.join(out, 'per_flight.csv'), index=False)
    wind_df.to_csv(os.path.join(out, 'wind.csv'), index=False)

    # coverage vs nominal level, from the |error| / sigma ratios of the scored steps
    nominal = np.linspace(.01, .99, 99)
    labels = list(dict.fromkeys(label for label, _ in curves))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, sigma in zip(axes, ['raw', 'recalibrated']):
        ax.plot([0, 100], [0, 100], color='grey', lw=1, ls='--', label='ideal')
        for label in labels:
            ratio = curves[label, sigma]
            ls = '--' if args.split != 'random' and not label.endswith('({})'.format(args.split)) else '-'
            ax.plot(nominal * 100, [np.mean(ratio <= norm.ppf(.5 + q / 2)) * 100 for q in nominal], ls=ls, label=label)
        ax.set(title='{} sigma'.format(sigma), xlabel='nominal interval (%)', xlim=(0, 100), ylim=(0, 100))
        ax.grid(alpha=.3)
    axes[0].set_ylabel('scored steps inside the interval (%)')
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'calibration.png'), dpi=150)

    fig, axes = plt.subplots(1, len(labels), figsize=(4 * len(labels), 3.8), squeeze=False)
    for ax, (label, pf) in zip(axes[0], per_flight.groupby('label' if 'label' in per_flight else 'method', sort=False)):
        ax.scatter(pf['wind'], pf['rmse'], s=14, label='RMSE')
        ax.scatter(pf['wind'], pf['mean_sigma'], s=14, label='mean sigma (raw)')
        ax.axvline(WIND_THRESHOLD, color='grey', lw=1, ls='--')
        ax.set(title=label, xlabel='cruise wind (m/s)', ylabel='W')
        ax.grid(alpha=.3)
    axes[0][0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'wind.png'), dpi=150)

    pd.set_option('display.width', 220)
    print('\ncalibration on {} flights (coverage in % of steps; energy coverage in % of flights):'.format(len(test)))
    print(calib.round(2).to_string(index=False))
    print('\nwind ({:.2f} m/s threshold):'.format(WIND_THRESHOLD))
    print(wind_df.round(3).to_string(index=False))
    print('\nsaved', out)
