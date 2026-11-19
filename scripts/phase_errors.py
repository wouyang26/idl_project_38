"""Where in the flight does the wind error happen? Power error per flight phase, held-out model vs its control.

    python scripts/phase_errors.py                       # C checkpoints -> Results/phase_errors/

Phases from altitude (diffalt), per flight: climb until the drone first reaches 90 % of its peak altitude, descent
after the last step above it, cruise in between. Flights: the 20 unseen windy flights of `wind` /
`random_sub_wind`, and the 11 calm flights of `windy_days` / `random_sub_windy_days` as a calm reference. Per phase:
MAPE and the signed bias (mean of predicted - measured, W; negative = the model under-predicts power), mean over the
flights per seed, then mean / std over seeds. Writes per_flight.csv and summary.csv.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from data_utils import load_dataset, process_data  # noqa: E402
from fit_liu import unseen_test  # noqa: E402
from splits import load_conditions  # noqa: E402
from stress_test import FIRST  # noqa: E402
from torch_data import create_tensors, predict_flight  # noqa: E402
from torch_model import load_model  # noqa: E402

PHASES = ['climb', 'cruise', 'descent']


def phases(alt):
    """'climb' / 'cruise' / 'descent' per step, from altitude."""
    high = np.flatnonzero(alt >= 0.9 * alt.max())
    label = np.full(len(alt), 'cruise', dtype=object)
    label[:high[0]] = 'climb'
    label[high[-1] + 1:] = 'descent'
    return label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', default='./', help='Working directory for data.')
    parser.add_argument('--ckpts', default='Results/ckpts/tcn_20_*_2_1_5_*wind*/final.pt', help='Glob of checkpoints.')
    parser.add_argument('--out', default='Results/phase_errors', help='Output directory.')
    args = parser.parse_args()

    data_dir = os.path.join(args.directory, 'data/')
    all_data = load_dataset(data_dir)
    cond = load_conditions(data_dir)
    test = {s: unseen_test(all_data, s, data_dir, cond) for s in ['wind', 'windy_days']}
    test.update({'random_sub_wind': test['wind'], 'random_sub_windy_days': test['windy_days']})
    labels = {f: phases(all_data[f]['diffalt'].values[FIRST:]) for f in set(test['wind']) | set(test['windy_days'])}

    rows = []
    for path in sorted(glob.glob(args.ckpts)):
        model, inputs, ckpt = load_model(path)
        if inputs['split'] not in test or inputs.get('variance_head'):
            continue
        flights = test[inputs['split']]
        data, data_min, data_max, _, _, _ = process_data(all_data, stats=(ckpt['data_min'], ckpt['data_max']), **inputs)
        tensors = create_tensors(data, flights, **inputs)
        size = inputs['nb_filters'] if inputs['net_type'] in ('tcn', 'residual') else inputs['units']
        arch = '{}{}{}'.format(inputs['net_type'], size, '' if inputs.get('head_act', 'relu') == 'relu' else '-leaky')
        for f in flights:
            pred = predict_flight(model, *tensors[f][:2], inputs['batch_size'], inputs['stateful'])
            pred = pred * (data_max[-1] - data_min[-1]) + data_min[-1]
            true = all_data[f]['power'].values[FIRST:]
            for ph in PHASES:
                m = labels[f] == ph
                if m.sum() < 5:
                    continue
                rows.append({'run': os.path.basename(os.path.dirname(path)), 'arch': arch, 'split': inputs['split'],
                             'seed': ckpt['seed'], 'flight': f, 'wind': cond.loc[f, 'wind'], 'phase': ph,
                             'share': m.mean(), 'mape': np.mean(np.abs((true[m] - pred[m]) / true[m])) * 100,
                             'bias_w': np.mean(pred[m] - true[m])})
        print('done', os.path.basename(os.path.dirname(path)), flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, 'per_flight.csv'), index=False)
    per_seed = df.groupby(['arch', 'split', 'phase', 'seed'])[['share', 'mape', 'bias_w']].mean()
    summary = per_seed.groupby(['arch', 'split', 'phase']).agg(['mean', 'std']).round(2)
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.to_csv(os.path.join(args.out, 'summary.csv'))
    print(summary.to_string())
    print('saved', args.out)
