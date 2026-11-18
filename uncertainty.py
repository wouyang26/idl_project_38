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

All checkpoints must be trained on the `random` split; they are scored on its test flights. Each method is also
recalibrated: sigma is scaled by the factor that makes the 90 % interval cover 90 % of the validation steps.

Outputs in --out:
    calibration.csv     per method (raw / recalibrated): MAPE, coverage of the 50/80/90/95 % intervals, mean 90 %
                        width, NLL, and the same for the flight energy
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
from splits import WIND_THRESHOLD, load_conditions
from torch_data import create_tensors, predict_flight
from torch_model import load_model

LEVELS = [.5, .8, .9, .95]
FIRST = 19  # lookback - 1: the first predicted step


def member_paths(prefix, seeds, ckpt_dir):
    return [os.path.join(ckpt_dir, prefix + ('' if s == 0 else '_s{}'.format(s)), 'final.pt') for s in seeds]


def method_paths(prefix, ckpt_dir):
    """{method: (checkpoints, MC samples)} for the methods whose checkpoints exist."""
    methods = {
        'ensemble': (member_paths(prefix, [0, 1, 2], ckpt_dir), 0),
        'var_head': (member_paths(prefix + '_var', [0], ckpt_dir), 0),
        'var_ensemble': (member_paths(prefix + '_var', [0, 1, 2], ckpt_dir), 0),
        'mc_dropout': (member_paths(prefix + '_d0.1', [0], ckpt_dir), None),
    }
    return {m: v for m, v in methods.items() if all(os.path.exists(p) for p in v[0])}


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
    row['energy_sd_kJ'] = np.mean(e_sd)
    return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', default='./', help='Working directory for data.')
    parser.add_argument('--prefix', default='tcn_20_64_2_1_5', help='Run name of the seed-0 plain model.')
    parser.add_argument('--ckpt_dir', default='Results/ckpts')
    parser.add_argument('--mc_samples', default=30, type=int, help='Forward passes for MC dropout.')
    parser.add_argument('--out', default='Results/uncertainty')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    data_dir = os.path.join(args.directory, 'data/')
    all_data = load_dataset(data_dir)
    _, _, _, test_range, val_range, _ = get_flights_and_ranges(all_data)
    test, val = sorted(set(int(f) for f in test_range)), sorted(set(int(f) for f in val_range))
    wind = load_conditions(data_dir)['wind']

    methods = method_paths(args.prefix, args.ckpt_dir)
    if not methods:
        raise SystemExit('no checkpoints found for prefix {!r} in {}'.format(args.prefix, args.ckpt_dir))
    for name, (paths, _) in methods.items():
        for p in paths:  # every method must be scored on flights none of its members trained or validated on
            ranges = torch.load(p, map_location='cpu', weights_only=True).get('ranges')
            if ranges is not None and set(ranges['train'] + ranges['val']) & set(test):
                raise SystemExit('{} trained on test flights; uncertainty.py needs random-split runs'.format(p))

    truth = {f: (all_data[f]['power'].values[FIRST:], all_data[f]['time'].values[FIRST:]) for f in test + val}
    os.makedirs(args.out, exist_ok=True)
    rows, flight_rows, wind_rows, curves = [], [], [], {}
    for name, (paths, mc) in methods.items():
        mc = args.mc_samples if mc is None else mc
        print('{}: {} checkpoint(s){}'.format(name, len(paths), ', {} MC passes'.format(mc) if mc else ''))
        preds = member_predictions(paths, all_data, test + val, args.device, mc)
        dist = {f: gaussian(*preds[f]) for f in preds}
        energy = {f: energy_gaussian(*preds[f], truth[f][1]) for f in preds}
        e_true = {f: np.trapezoid(*truth[f]) / 1000 for f in preds}

        # recalibrate on the validation flights: one factor for the steps, one for the flight energy
        s = scale_factor(np.concatenate([truth[f][0] for f in val]), np.concatenate([dist[f][0] for f in val]),
                         np.concatenate([dist[f][1] for f in val]))
        z_val = [(e_true[f] - energy[f][0]) / energy[f][1] for f in val]
        s_e = np.quantile(np.abs(z_val), .9) / norm.ppf(.95)

        mape = np.mean([np.mean(np.abs((truth[f][0] - dist[f][0]) / truth[f][0])) * 100 for f in test])
        for label, k, k_e in [('raw', 1., 1.), ('recalibrated', s, s_e)]:
            row = {'method': name, 'sigma': label, 'scale': k, 'energy_scale': k_e, 'mape': mape}
            row.update(metrics([truth[f][0] for f in test], [dist[f][0] for f in test], [dist[f][1] * k for f in test],
                               [e_true[f] for f in test], [energy[f][0] for f in test],
                               [energy[f][1] * k_e for f in test]))
            rows.append(row)
            curves[name, label] = (np.concatenate([np.abs(truth[f][0] - dist[f][0]) / np.maximum(dist[f][1] * k, 1e-9)
                                                   for f in test]))

        per_flight = pd.DataFrame([{
            'method': name, 'flight': f, 'wind': wind.get(f, np.nan),
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
            'method': name, 'n_calm': int((~windy).sum()), 'n_windy': int(windy.sum()),
            'spearman_wind_sigma': spearmanr(pf['wind'], pf['mean_sigma'])[0],
            'spearman_wind_rmse': spearmanr(pf['wind'], pf['rmse'])[0],
            'sigma_calm': pf.loc[~windy, 'mean_sigma'].mean(), 'sigma_windy': pf.loc[windy, 'mean_sigma'].mean(),
            'rmse_calm': pf.loc[~windy, 'rmse'].mean(), 'rmse_windy': pf.loc[windy, 'rmse'].mean(),
            'cov90_recal_calm': pf.loc[~windy, 'cov90_recal'].mean(),
            'cov90_recal_windy': pf.loc[windy, 'cov90_recal'].mean(),
        })

    calib = pd.DataFrame(rows)
    per_flight = pd.concat(flight_rows)
    wind_df = pd.DataFrame(wind_rows)
    calib.to_csv(os.path.join(args.out, 'calibration.csv'), index=False)
    per_flight.to_csv(os.path.join(args.out, 'per_flight.csv'), index=False)
    wind_df.to_csv(os.path.join(args.out, 'wind.csv'), index=False)

    # coverage vs nominal level, from the |error| / sigma ratios of the test steps
    nominal = np.linspace(.01, .99, 99)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, label in zip(axes, ['raw', 'recalibrated']):
        ax.plot([0, 100], [0, 100], color='grey', lw=1, ls='--', label='ideal')
        for name in methods:
            ratio = curves[name, label]
            ax.plot(nominal * 100, [np.mean(ratio <= norm.ppf(.5 + q / 2)) * 100 for q in nominal], label=name)
        ax.set(title='{} sigma'.format(label), xlabel='nominal interval (%)', xlim=(0, 100), ylim=(0, 100))
        ax.grid(alpha=.3)
    axes[0].set_ylabel('test steps inside the interval (%)')
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'calibration.png'), dpi=150)

    fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 3.8), squeeze=False)
    for ax, (name, pf) in zip(axes[0], per_flight.groupby('method', sort=False)):
        ax.scatter(pf['wind'], pf['rmse'], s=14, label='RMSE')
        ax.scatter(pf['wind'], pf['mean_sigma'], s=14, label='mean sigma (raw)')
        ax.axvline(WIND_THRESHOLD, color='grey', lw=1, ls='--')
        ax.set(title=name, xlabel='cruise wind (m/s)', ylabel='W')
        ax.grid(alpha=.3)
    axes[0][0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'wind.png'), dpi=150)

    pd.set_option('display.width', 200)
    print('\ncalibration on {} test flights (coverage in % of steps; energy coverage in % of flights):'.format(len(test)))
    print(calib.round(2).to_string(index=False))
    print('\nwind ({:.2f} m/s threshold):'.format(WIND_THRESHOLD))
    print(wind_df.round(3).to_string(index=False))
    print('\nsaved', args.out)
