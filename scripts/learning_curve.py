"""Experiment A3: test error vs number of training flights, physics + TCN hybrid vs plain TCN (and refit Liu).

    python scripts/learning_curve.py                     # Results/stress_test/per_flight.csv -> Results/learning_curve/
    python scripts/learning_curve.py --per_flight <csv> --out docs/learning_curve

Reads the per-flight errors that stress_test.py wrote for the `--train_size N --subset_seed k` runs (random split),
scores them on the random split's test flights, and averages over flights, then mean / std over the training
subsets. Full-size random-split runs of the same models (no --train_size) are added as N = 114. Liu, refit on each
same training subset, is the physics-only reference. Writes learning_curve.csv and learning_curve.png.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from data_utils import find_regimes, load_dataset  # noqa: E402
from fit_liu import score  # noqa: E402
from physics_torch import fit_liu  # noqa: E402
from splits import get_split, subsample_train  # noqa: E402
from stress_test import FIRST  # noqa: E402

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', default='./', help='Working directory for data.')
    parser.add_argument('--per_flight', default='Results/stress_test/per_flight.csv', help='From stress_test.py.')
    parser.add_argument('--out', default='Results/learning_curve', help='Output directory.')
    args = parser.parse_args()

    data_dir = os.path.join(args.directory, 'data/')
    all_data = load_dataset(data_dir)
    ranges = get_split(all_data, 'random', data_dir)
    test = sorted(set(ranges[0]))

    df = pd.read_csv(args.per_flight)
    df = df[(df['split'] == 'random') & df['flight'].isin(test) & (df['arch'] != 'liu')]
    curve = df[df['train_size'].notna()]
    if curve.empty:
        sys.exit('no --train_size runs in {}'.format(args.per_flight))
    full = df[df['train_size'].isna() & df['arch'].isin(curve['arch']) & (df['seed'] == 0)]
    full = full.assign(train_size=len(ranges[2]), subset_seed=0)
    runs = pd.concat([curve, full]).groupby(['arch', 'train_size', 'subset_seed', 'run'])[['mape', 're']].mean()

    # Liu refit on exactly the same training subsets (no network), as the physics-only reference
    cps = find_regimes(all_data, FIRST)
    liu = []
    for (n, k) in sorted(set(zip(runs.index.get_level_values('train_size'), runs.index.get_level_values('subset_seed')))):
        train = subsample_train(ranges, int(n), int(k))[2] if n < len(ranges[2]) else ranges[2]
        mape, re = score(all_data, test, cps, fit_liu(all_data, train))
        liu.append({'arch': 'liu-refit', 'train_size': n, 'subset_seed': k, 'mape': mape, 're': re})
    runs = pd.concat([runs.reset_index(), pd.DataFrame(liu)])

    g = runs.groupby(['arch', 'train_size'])[['mape', 're']].agg(['mean', 'std', 'count'])
    g.columns = ['_'.join(c) for c in g.columns]
    g = g.reset_index()
    os.makedirs(args.out, exist_ok=True)
    g.to_csv(os.path.join(args.out, 'learning_curve.csv'), index=False)
    print(g.round(2).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, m, title in zip(axes, ['mape', 're'], ['power MAPE (%)', 'energy RE (%)']):
        for arch, a in g.groupby('arch'):
            ax.errorbar(a['train_size'], a[m + '_mean'], a[m + '_std'].fillna(0), marker='o', capsize=3,
                        ls='--' if arch == 'liu-refit' else '-', label=arch)
        ax.set_xscale('log')
        ax.set_xticks(sorted(g['train_size'].unique()))
        ax.set_xticklabels([int(n) for n in sorted(g['train_size'].unique())])
        ax.set_xlabel('training flights')
        ax.set_title(title + ' on the 35 random-split test flights')
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'learning_curve.png'), dpi=150)
    print('saved', args.out)
