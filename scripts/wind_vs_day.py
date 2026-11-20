"""Is the wind gap a wind effect or a "different day" effect? Reads the per-flight errors written by stress_test.py.

    python scripts/wind_vs_day.py                                    # Results/stress_test/per_flight.csv
    python scripts/wind_vs_day.py --per_flight Results/stress_test/per_flight.csv --out docs/wind_vs_day

1. wind vs random_sub_wind (no new training): the unseen windy test flights, split into those on windy days (few
   flights of that day are in the wind model's training) and those on other days (their days are trained on). A gap
   that stays on the other days is wind, not the day.
2. windy_days vs random_sub_windy_days: the calm flights of the windy days. The held-out model never trained on
   those days, the control trained on all their other flights, and neither saw much wind on the test flights, so
   this gap is the day effect alone.
Mean / std over seeds per model -> <out>/wind_vs_day.csv.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from data_utils import load_dataset  # noqa: E402
from splits import get_split, load_conditions, windy_days  # noqa: E402
from stress_test import control_of  # noqa: E402


def gaps(df, split, flights):
    """Per model: held-out and control MAPE / RE on `flights`, mean over flights per seed, then mean / std."""
    rows = []
    for arch in sorted(set(df['arch']) - {'liu'}):
        d = df[(df['arch'] == arch) & df['flight'].isin(flights)]
        row = {'arch': arch}
        for role, s in [('heldout', split), ('control', control_of(split))]:
            per_seed = d[d['split'] == s].groupby('seed')[['mape', 're']].mean()
            row[role + '_seeds'] = len(per_seed)
            for m in ['mape', 're']:
                row['{}_{}'.format(role, m)] = per_seed[m].mean()
                row['{}_{}_std'.format(role, m)] = per_seed[m].std()
        if row['heldout_seeds'] and row['control_seeds']:
            for m in ['mape', 're']:
                row['gap_' + m] = row['heldout_' + m] - row['control_' + m]
            rows.append(row)
    return rows


def unseen(df, split, test):
    """Test flights that no run of `split` or of its control trained or validated on."""
    pair = df[df['split'].isin([split, control_of(split)])]
    return sorted(set(test) - set(pair.loc[pair['seen'], 'flight']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', default='./', help='Working directory for data.')
    parser.add_argument('--per_flight', default='Results/stress_test/per_flight.csv', help='From stress_test.py.')
    parser.add_argument('--out', default='Results/wind_vs_day', help='Output directory.')
    args = parser.parse_args()

    data_dir = os.path.join(args.directory, 'data/')
    all_data = load_dataset(data_dir)
    cond = load_conditions(data_dir)
    days = windy_days(cond)
    df = pd.read_csv(args.per_flight)
    print('windy days:', ', '.join(days))

    rows = []
    wind_test = unseen(df, 'wind', get_split(all_data, 'wind', data_dir, cond=cond)[0])
    on_day = cond.loc[wind_test, 'date'].isin(days).values
    for group, fs in [('all', wind_test), ('on windy days', np.array(wind_test)[on_day]),
                      ('on other days', np.array(wind_test)[~on_day])]:
        rows += [dict(r, comparison='wind', flights=group, n=len(fs)) for r in gaps(df, 'wind', fs)]
    calm_test = unseen(df, 'windy_days', get_split(all_data, 'windy_days', data_dir, cond=cond)[0])
    rows += [dict(r, comparison='windy_days', flights='calm, on windy days', n=len(calm_test))
             for r in gaps(df, 'windy_days', calm_test)]
    if not rows:
        sys.exit('no runs of wind / windy_days and their controls in {}'.format(args.per_flight))

    g = pd.DataFrame(rows)
    first = ['arch', 'comparison', 'flights', 'n']
    g = g[first + [c for c in g.columns if c not in first]]
    os.makedirs(args.out, exist_ok=True)
    g.to_csv(os.path.join(args.out, 'wind_vs_day.csv'), index=False)
    cols = first + ['heldout_mape', 'control_mape', 'gap_mape', 'gap_re', 'heldout_seeds']
    print(g[[c for c in cols if c in g.columns]].round(2).to_string(index=False))
    print('saved', os.path.join(args.out, 'wind_vs_day.csv'))
