"""PyTorch version of `evaluation.py` for the deep models: power MAPE / energy RE per flight set.

    python evaluate_torch.py                                  # b-TCN on the test flights
    python evaluate_torch.py -m all -e random test            # README table
    python evaluate_torch.py -m Results/ckpts/<save_str>/final.pt
    python evaluate_torch.py -m s-TCN Results/ckpts/tcn_20_16_2_1_5_speed12/final.pt -e speed12   # held-out 12 m/s flights
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from scipy.integrate import simpson
from tabulate import tabulate

from data_utils import find_regimes, get_flights_and_ranges, get_random_flights, load_dataset, process_data
from splits import SPLITS, get_split, load_conditions
from torch_data import create_tensors, predict_flight
from torch_model import PRETRAINED_CONFIGS, load_model

import liu_model


def get_flights(all_data, data_dir='./data/'):
    """Flight sets by name: the paper's train/val/test/random, plus the test flights of each held-out split."""
    _, _, _, test_range, val_range, train_range = get_flights_and_ranges(all_data)
    flights = {'train': train_range, 'test': test_range, 'val': val_range, 'random': get_random_flights()}
    cond = load_conditions(data_dir)
    for split in SPLITS[1:]:
        flights[split] = get_split(all_data, split, data_dir, cond=cond)[0]
    return flights


def simps_avg(y, x):
    """scipy's old `simps(y, x, even='avg')` (removed in scipy 1.14), used for the paper's numbers."""
    if len(y) % 2 == 1 or len(y) < 3:
        return simpson(y, x=x) if len(y) >= 3 else np.trapezoid(y, x)
    first = simpson(y[:-1], x=x[:-1]) + .5 * (x[-1] - x[-2]) * (y[-1] + y[-2])
    last = simpson(y[1:], x=x[1:]) + .5 * (x[1] - x[0]) * (y[1] + y[0])
    return (first + last) / 2


def calc_energy(power, ts, cp):
    """Energy (kJ) of each regime of a flight; `cp` holds regime start indices + last index."""
    assert len(power) == len(ts)
    assert cp[-1] == len(power)
    return np.array([simps_avg(power[c:cp[i + 1]], ts[c:cp[i + 1]]) / 1000 for i, c in enumerate(cp[:-1])])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', default='./', help='Working directory for data.')
    parser.add_argument('-m', '--model', default=['b-TCN'], nargs='*',
                        help='Pretrained name(s) (b-TCN, s-TCN, b-LSTM, s-LSTM, liu, all) or path(s) to .pt checkpoints.')
    parser.add_argument('--models_dir', default='./models_torch', help='Where port_weights.py saved the pretrained models.')
    parser.add_argument('-e', '--evaluate', default=['test'], nargs='*', choices=['train', 'test', 'val', 'random', 'all'] + SPLITS[1:],
                        help='Flight sets; a held-out split name (e.g. speed12) means that split\'s test flights.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--unseen', action='store_true',
                        help='Only score flights that none of the models trained or validated on (same flights in every row).')
    args = parser.parse_args()

    if 'all' in args.model:
        models = list(PRETRAINED_CONFIGS)
        models.append('liu')
    else:
        models = args.model
    datasets = ['train', 'test', 'val', 'random'] if 'all' in args.evaluate else list(dict.fromkeys(args.evaluate))

    all_data = load_dataset(os.path.join(args.directory, 'data/'))
    flight_ranges = get_flights(all_data, os.path.join(args.directory, 'data/'))
    eval_range = list(np.unique(np.concatenate([flight_ranges[d] for d in datasets])))

    first = 19  # lookback - 1
    change_points = find_regimes({k: all_data[k] for k in eval_range}, first)
    truth = {
        k: (all_data[k]['power'].values[first:],
            calc_energy(all_data[k]['power'].values[first:], all_data[k]['time'].values[first:], change_points[k]))
        for k in eval_range
    }

    results, seen = [], set()
    for name in models:
        if name == 'liu':
            [k1, k2, c2, c4, c5] = liu_model.optimum_values()
        else:
            path = name if os.path.exists(name) else os.path.join(args.models_dir, name + '.pt')
            model, inputs, ckpt = load_model(path, map_location=args.device)
            # normalize with the stats the model was trained with (the ported pretrained models don't store them,
            # but were trained on the random split, which is what process_data recomputes by default)
            stats = (ckpt['data_min'], ckpt['data_max']) if 'data_min' in ckpt else None
            data, data_min, data_max, _, _, _ = process_data(all_data, stats=stats, **inputs)
            tensors = create_tensors(data, eval_range, device=args.device, **inputs)
            if 'ranges' in ckpt:
                seen.update(ckpt['ranges']['train'] + ckpt['ranges']['val'])
            else:  # ported pretrained models were trained on the paper's random split
                seen.update(flight_ranges['train'])
                seen.update(flight_ranges['val'])

        mape, re = {}, {}
        for flight in eval_range:
            if name == 'liu':
                liu_dataset = all_data[flight][['vertspd', 'airspeed', 'aoa', 'payload', 'density']].T
                y_pred = liu_model.power(liu_dataset, k1, k2, c2, c4, c5)[first:]
            else:
                y_pred = predict_flight(model, *tensors[flight][:2], inputs['batch_size'], inputs['stateful'])
                y_pred = y_pred * (data_max[-1] - data_min[-1]) + data_min[-1]
            true_w, true_e = truth[flight]
            mape[flight] = np.mean(np.abs((true_w - y_pred) / true_w)) * 100
            pred_e = calc_energy(y_pred, all_data[flight]['time'].values[first:], change_points[flight])
            re[flight] = np.mean(np.abs((true_e - pred_e) / true_e)) * 100

        results.append((name, mape, re))

    if args.unseen:
        flight_ranges = {d: [f for f in flight_ranges[d] if f not in seen] for d in datasets}
    rows = []
    for name, mape, re in results:
        row = {'Model': name}
        for d in datasets:
            fs = flight_ranges[d]
            col = '{} (n={})'.format(d, len(set(fs))) if args.unseen else d
            row[col] = '{:.2f}/{:.2f}'.format(np.mean([mape[f] for f in fs]),
                                             np.mean([re[f] for f in fs])) if fs else '-'
        rows.append(row)

    print('MAPE (%) / RE (%)')
    print(tabulate(pd.DataFrame(rows), headers='keys', tablefmt='fancy_grid', showindex=False))
