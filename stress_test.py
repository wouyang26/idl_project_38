"""Stress test: each held-out split vs its data-size control, scored on flights neither model saw.

    python stress_test.py                                   # all of Results/ckpts -> Results/stress_test/
    python stress_test.py --ckpts 'Results/ckpts/tcn_20_16_2_1_5*/final.pt'

1. Per-flight MAPE / RE of every checkpoint (and Liu) on every flight -> per_flight.csv. Runs already in the csv
   are not re-evaluated, so new checkpoints can be added later.
2. For each model and held-out split: the flights that no run of the split or of its control (see
   `splits.CONTROLS`, default random_subsampled) trained or validated on; mean over those flights per seed, then
   mean / std over seeds -> gaps.csv.
3. gaps.png (held-out minus control per split) and speed_<model>.png (MAPE by test speed for the speed splits).
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import liu_model
from data_utils import find_regimes, load_dataset, process_data
from evaluate_torch import calc_energy
from splits import CONTROLS, SPLITS, get_split, load_conditions
from torch_data import create_tensors, predict_flight
from torch_model import load_model

FIRST = 19  # lookback - 1, as in evaluate_torch.py
SPEED_SPLITS = ['speed12', 'speed10-12', 'speed8-12']


def control_of(split):
    """The data-size control a held-out split is compared with."""
    return next((c for c, s in CONTROLS.items() if s == split), 'random_subsampled')


def per_flight(paths, all_data, device, liu=True):
    """One row per (run, flight): MAPE / RE in % and whether the run trained or validated on the flight."""
    flights = sorted(all_data)
    cps = find_regimes(all_data, FIRST)
    truth = {}
    for f in flights:
        power, ts = all_data[f]['power'].values[FIRST:], all_data[f]['time'].values[FIRST:]
        truth[f] = (power, calc_energy(power, ts, cps[f]))

    rows = []
    for path in paths + (['liu'] if liu else []):
        if path == 'liu':
            k1, k2, c2, c4, c5 = liu_model.optimum_values()
            run, seen = {'run': 'liu', 'arch': 'liu', 'split': None, 'seed': 0}, set()
        else:
            model, inputs, ckpt = load_model(path, map_location=device)
            data, data_min, data_max, _, _, _ = process_data(all_data, stats=(ckpt['data_min'], ckpt['data_max']), **inputs)
            tensors = create_tensors(data, flights, device=device, **inputs)
            size = inputs['nb_filters'] if inputs['net_type'] == 'tcn' else inputs['units']
            run = {'run': os.path.basename(os.path.dirname(path)), 'arch': '{}{}'.format(inputs['net_type'], size),
                   'split': inputs['split'], 'seed': ckpt['seed']}
            seen = set(ckpt['ranges']['train'] + ckpt['ranges']['val'])
        for f in flights:
            if path == 'liu':
                y_pred = liu_model.power(all_data[f][['vertspd', 'airspeed', 'aoa', 'payload', 'density']].T,
                                         k1, k2, c2, c4, c5)[FIRST:]
            else:
                y_pred = predict_flight(model, *tensors[f][:2], inputs['batch_size'], inputs['stateful'])
                y_pred = y_pred * (data_max[-1] - data_min[-1]) + data_min[-1]
            true_w, true_e = truth[f]
            pred_e = calc_energy(y_pred, all_data[f]['time'].values[FIRST:], cps[f])
            with np.errstate(divide='ignore', invalid='ignore'):  # flight 232 has power <= 0 W
                rows.append(dict(run, flight=f, seen=f in seen,
                                 mape=np.mean(np.abs((true_w - y_pred) / true_w)) * 100,
                                 re=np.mean(np.abs((true_e - pred_e) / true_e)) * 100))
        print('evaluated', run['run'])
    return pd.DataFrame(rows)


def compare(df, all_data, cond, data_dir):
    """(gap table, {(arch, split): flights}) for every held-out split that has both its own and its control's runs."""
    rows, flights = [], {}
    for arch in sorted(set(df['arch']) - {'liu'}):
        d = df[df['arch'] == arch]
        for split in [s for s in SPLITS[1:] if s not in CONTROLS]:
            control = control_of(split)
            pair = d[d['split'].isin([split, control])]
            if set(pair['split']) != {split, control}:
                continue  # this model hasn't been trained on both yet
            test = set(get_split(all_data, split, data_dir, cond=cond)[0])
            fs = sorted(test - set(pair.loc[pair['seen'], 'flight']))
            flights[arch, split] = fs
            row = {'arch': arch, 'split': split, 'control': control, 'n': len(fs)}
            for role, s in [('heldout', split), ('control', control)]:
                per_seed = pair[(pair['split'] == s) & pair['flight'].isin(fs)].groupby('seed')[['mape', 're']].mean()
                row[role + '_seeds'] = len(per_seed)
                for m in ['mape', 're']:
                    row['{}_{}'.format(role, m)] = per_seed[m].mean()
                    row['{}_{}_std'.format(role, m)] = per_seed[m].std()
            liu = df[(df['arch'] == 'liu') & df['flight'].isin(fs)]
            for m in ['mape', 're']:
                row['gap_' + m] = row['heldout_' + m] - row['control_' + m]
                row['liu_' + m] = liu[m].mean()
            rows.append(row)
    return pd.DataFrame(rows), flights


def plot_gaps(g, out):
    """Held-out minus control per split and model; error bars combine the two seed stds."""
    splits = list(dict.fromkeys(g['split']))
    archs = sorted(set(g['arch']))
    w = 0.8 / len(archs)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, m, title in zip(axes, ['mape', 're'], ['power MAPE', 'energy RE']):
        for i, arch in enumerate(archs):
            a = g[g['arch'] == arch]
            x = np.array([splits.index(s) for s in a['split']]) + (i - (len(archs) - 1) / 2) * w
            err = np.sqrt(a['heldout_{}_std'.format(m)] ** 2 + a['control_{}_std'.format(m)] ** 2).fillna(0)
            ax.bar(x, a['gap_' + m], w, yerr=err, capsize=3, label=arch)
        ax.axhline(0, color='k', lw=0.8)
        ax.set_xticks(range(len(splits)))
        ax.set_xticklabels(splits, rotation=30, ha='right')
        ax.set_ylabel('held-out minus control (% points)')
        ax.set_title(title)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'gaps.png'), dpi=150)


def plot_speed(df, flights, cond, out):
    """MAPE by test speed for the speed splits (solid) and their controls (dashed), on the same flights."""
    for arch in sorted({a for a, s in flights if s in SPEED_SPLITS}):
        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        for i, split in enumerate(SPEED_SPLITS):
            if (arch, split) not in flights:
                continue
            for s, ls, label in [(split, '-', split), (control_of(split), '--', split + ' control')]:
                d = df[(df['arch'] == arch) & (df['split'] == s) & df['flight'].isin(flights[arch, split])]
                d = d.assign(speed=cond.loc[d['flight'], 'speed'].values)
                by_speed = d.groupby(['speed', 'seed'])['mape'].mean().groupby('speed')
                mean = by_speed.mean()
                ax.errorbar(mean.index + 0.1 * i, mean, by_speed.std().fillna(0), color='C{}'.format(i), ls=ls,
                            marker='o', capsize=3, label=label)
        ax.set_xlabel('test speed (m/s)')
        ax.set_ylabel('MAPE (%)')
        ax.set_title('{}: unseen flights by speed'.format(arch))
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'speed_{}.png'.format(arch)), dpi=150)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', default='./', help='Working directory for data.')
    parser.add_argument('--ckpts', default='Results/ckpts/*/final.pt', help='Glob of checkpoints to evaluate.')
    parser.add_argument('--out', default='Results/stress_test', help='Output directory.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data_dir = os.path.join(args.directory, 'data/')
    all_data = load_dataset(data_dir)
    cond = load_conditions(data_dir)

    csv = os.path.join(args.out, 'per_flight.csv')
    df = pd.read_csv(csv) if os.path.exists(csv) else None
    done = set() if df is None else set(df['run'])
    paths = [p for p in sorted(glob.glob(args.ckpts)) if os.path.basename(os.path.dirname(p)) not in done]
    if paths or 'liu' not in done:
        new = per_flight(paths, all_data, args.device, liu='liu' not in done)
        df = new if df is None else pd.concat([df, new], ignore_index=True)
        df.to_csv(csv, index=False)

    g, flights = compare(df, all_data, cond, data_dir)
    g.to_csv(os.path.join(args.out, 'gaps.csv'), index=False)
    cols = ['arch', 'split', 'n', 'heldout_mape', 'control_mape', 'gap_mape', 'heldout_re', 'control_re', 'gap_re',
            'liu_mape']
    print(g[cols].round(2).to_string(index=False))
    plot_gaps(g, args.out)
    plot_speed(df, flights, cond, args.out)
    print('saved', args.out)
