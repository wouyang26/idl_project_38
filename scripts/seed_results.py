"""Stress-test results of one model, kept apart per seed so the splits can be compared seed by seed.

    python scripts/seed_results.py --ckpts 'Results/ckpts/lstm_20_0.0_64_3_RMSprop*/final.pt' --out docs/stress_test_lstm64

Uses stress_test.py (same unseen flights, same metrics) and writes
    <out>/seed<k>/gaps.csv, gaps.png, speed_<arch>.png   held-out vs control for seed k alone
    <out>/all_seeds/...                                    the same over every seed (mean / std), if more than one
    <out>/per_seed.csv                                     one row per (split, seed), to compare splits across seeds
The per-flight errors are cached in --cache, so a new seed only evaluates its own checkpoints.
"""
import argparse
import glob
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from data_utils import load_dataset  # noqa: E402
from splits import load_conditions  # noqa: E402
from stress_test import compare, per_flight, plot_gaps, plot_speed  # noqa: E402

COLS = ['arch', 'split', 'control', 'seed', 'n', 'heldout_mape', 'control_mape', 'gap_mape',
        'heldout_re', 'control_re', 'gap_re', 'liu_mape', 'liu_re']


def write(g, df, flights, cond, out):
    os.makedirs(out, exist_ok=True)
    g.to_csv(os.path.join(out, 'gaps.csv'), index=False)
    plot_gaps(g, out)
    plot_speed(df, flights, cond, out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', default='./', help='Working directory for data.')
    parser.add_argument('--ckpts', required=True, help='Glob of checkpoints to evaluate.')
    parser.add_argument('--out', required=True, help='Output directory (e.g. docs/stress_test_lstm64).')
    parser.add_argument('--cache', default='Results/stress_test_lstm64', help='Where per_flight.csv is cached.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    data_dir = os.path.join(args.directory, 'data/')
    all_data = load_dataset(data_dir)
    cond = load_conditions(data_dir)

    os.makedirs(args.cache, exist_ok=True)
    csv = os.path.join(args.cache, 'per_flight.csv')
    df = pd.read_csv(csv) if os.path.exists(csv) else None
    done = set() if df is None else set(df['run'])
    paths = [p for p in sorted(glob.glob(args.ckpts)) if os.path.basename(os.path.dirname(p)) not in done]
    if paths or 'liu' not in done:
        new = per_flight(paths, all_data, args.device, liu='liu' not in done)
        df = new if df is None else pd.concat([df, new], ignore_index=True)
        df.to_csv(csv, index=False)

    liu = df['arch'] == 'liu'
    rows = []
    for seed in sorted(df.loc[~liu, 'seed'].unique()):
        sub = df[liu | (df['seed'] == seed)]
        g, flights = compare(sub, all_data, cond, data_dir)
        write(g, sub, flights, cond, os.path.join(args.out, 'seed{}'.format(seed)))
        rows.append(g.assign(seed=seed))
        print('seed', seed)
        print(g[COLS[:2] + COLS[4:]].round(2).to_string(index=False))

    if len(rows) > 1:
        g, flights = compare(df, all_data, cond, data_dir)
        write(g, df, flights, cond, os.path.join(args.out, 'all_seeds'))
    pd.concat(rows, ignore_index=True)[COLS].to_csv(os.path.join(args.out, 'per_seed.csv'), index=False)
    print('saved', args.out)
