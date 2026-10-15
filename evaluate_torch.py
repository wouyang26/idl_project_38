"""PyTorch version of `evaluation.py` for the deep models: power MAPE / energy RE per flight set.

    python evaluate_torch.py                                  # b-TCN on the test flights
    python evaluate_torch.py -m all -e random test            # README table
    python evaluate_torch.py -m Results/ckpts/<save_str>/final.pt
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from scipy.integrate import simpson
from tabulate import tabulate

from data_utils import find_regimes, get_flights_and_ranges, get_random_flights, load_dataset, process_data
from torch_data import create_tensors, predict_flight
from torch_model import PRETRAINED_CONFIGS, load_model


def get_flights(all_data):
    _, _, _, test_range, val_range, train_range = get_flights_and_ranges(all_data)
    return {'train': train_range, 'test': test_range, 'val': val_range, 'random': get_random_flights()}


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
                        help='Pretrained name(s) (b-TCN, s-TCN, b-LSTM, s-LSTM, all) or path(s) to .pt checkpoints.')
    parser.add_argument('--models_dir', default='./models_torch', help='Where port_weights.py saved the pretrained models.')
    parser.add_argument('-e', '--evaluate', default=['test'], nargs='*', choices=['train', 'test', 'val', 'random', 'all'])
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    models = list(PRETRAINED_CONFIGS) if 'all' in args.model else args.model
    datasets = ['train', 'test', 'val', 'random'] if 'all' in args.evaluate else list(dict.fromkeys(args.evaluate))

    all_data = load_dataset(os.path.join(args.directory, 'data/'))
    flight_ranges = get_flights(all_data)
    eval_range = list(np.unique(np.concatenate([flight_ranges[d] for d in datasets])))

    first = 19  # lookback - 1
    change_points = find_regimes({k: all_data[k] for k in eval_range}, first)
    truth = {
        k: (all_data[k]['power'].values[first:],
            calc_energy(all_data[k]['power'].values[first:], all_data[k]['time'].values[first:], change_points[k]))
        for k in eval_range
    }

    rows = []
    for name in models:
        path = name if os.path.exists(name) else os.path.join(args.models_dir, name + '.pt')
        model, inputs, _ = load_model(path, map_location=args.device)
        data, data_min, data_max, _, _, _ = process_data(all_data, **inputs)
        tensors = create_tensors(data, eval_range, device=args.device, **inputs)

        mape, re = {}, {}
        for flight in eval_range:
            y_pred = predict_flight(model, *tensors[flight][:2], inputs['batch_size'], inputs['stateful'])
            y_pred = y_pred * (data_max[-1] - data_min[-1]) + data_min[-1]
            true_w, true_e = truth[flight]
            mape[flight] = np.mean(np.abs((true_w - y_pred) / true_w)) * 100
            pred_e = calc_energy(y_pred, all_data[flight]['time'].values[first:], change_points[flight])
            re[flight] = np.mean(np.abs((true_e - pred_e) / true_e)) * 100

        row = {'Model': name}
        for d in datasets:
            row[d] = '{:.2f}/{:.2f}'.format(np.mean([mape[f] for f in flight_ranges[d]]),
                                           np.mean([re[f] for f in flight_ranges[d]]))
        rows.append(row)

    print('MAPE (%) / RE (%)')
    print(tabulate(pd.DataFrame(rows), headers='keys', tablefmt='fancy_grid', showindex=False))
