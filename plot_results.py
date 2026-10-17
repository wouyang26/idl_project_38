"""Figures and per-flight metrics for a trained checkpoint vs a pretrained model (README "Baseline results").

    python plot_results.py                                       # 30-epoch s-TCN baseline vs pretrained s-TCN
    python plot_results.py -c Results/ckpts/<run>/final.pt -p b-TCN --out docs/<run>

Writes to --out: training_curves.png, per_flight_mape.png, example_flights.png, per_flight_metrics.csv
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data_utils import find_regimes, load_dataset, process_data
from evaluate_torch import calc_energy, get_flights
from torch_data import create_tensors, predict_flight
from torch_model import load_model

FIRST = 19  # lookback - 1, as in evaluate_torch.py


def predict_watts(path, all_data, flights, device):
    """{flight: predicted power in W} for one checkpoint."""
    model, inputs, _ = load_model(path, map_location=device)
    data, data_min, data_max, _, _, _ = process_data(all_data, **inputs)
    tensors = create_tensors(data, flights, device=device, **inputs)
    return {f: predict_flight(model, *tensors[f][:2], inputs['batch_size'], inputs['stateful'])
            * (data_max[-1] - data_min[-1]) + data_min[-1] for f in flights}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', default='./', help='Working directory for data.')
    parser.add_argument('-c', '--ckpt', default='Results/ckpts/tcn_20_16_2_1_5/final.pt')
    parser.add_argument('--csv', default=None, help='Training csv (default: CsvResults next to the checkpoint run).')
    parser.add_argument('-p', '--pretrained', default='s-TCN', help='Pretrained model name in models_torch/.')
    parser.add_argument('--out', default='docs/baseline')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    run = os.path.basename(os.path.dirname(args.ckpt))
    csv_path = args.csv or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(args.ckpt))),
                                        'CsvResults', run, 'csv_result.csv')

    # 1. training curves
    hist = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, metric in zip(axes, ['Loss', 'MAPE']):
        ax.plot(hist['Epoch'], hist['Training ' + metric], label='train')
        ax.plot(hist['Epoch'], hist['Validation ' + metric], label='validation')
        ax.set_xlabel('epoch')
        ax.set_ylabel(metric + (' (MSE, normalized power)' if metric == 'Loss' else ' (%, normalized power)'))
        ax.grid(alpha=.3)
        ax.legend()
    axes[0].set_yscale('log')
    fig.suptitle('Training curves: ' + run)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'training_curves.png'), dpi=130)
    plt.close(fig)

    # 2. per-flight power MAPE / energy RE on the test flights (random flights are a subset of test)
    all_data = load_dataset(os.path.join(args.directory, 'data/'))
    ranges = get_flights(all_data)
    random_set = set(ranges['random'])
    flights = sorted(set(ranges['test']))
    change_points = find_regimes({k: all_data[k] for k in flights}, FIRST)
    models = {'pretrained ' + args.pretrained: os.path.join('models_torch', args.pretrained + '.pt'),
              'ours (' + run + ')': args.ckpt}
    preds = {name: predict_watts(path, all_data, flights, args.device) for name, path in models.items()}

    rows = []
    for f in flights:
        true_w = all_data[f]['power'].values[FIRST:]
        ts = all_data[f]['time'].values[FIRST:]
        true_e = calc_energy(true_w, ts, change_points[f])
        row = {'flight': f, 'random': f in random_set, 'duration_s': ts[-1] - ts[0], 'energy_kJ': true_e.sum()}
        for name in models:
            pred_e = calc_energy(preds[name][f], ts, change_points[f])
            row[name + ' MAPE'] = np.mean(np.abs((true_w - preds[name][f]) / true_w)) * 100
            row[name + ' RE'] = np.mean(np.abs((true_e - pred_e) / true_e)) * 100
        rows.append(row)
    df = pd.DataFrame(rows)
    df.round(3).to_csv(os.path.join(args.out, 'per_flight_metrics.csv'), index=False)

    names = list(models)
    ours = names[1]
    order = df.sort_values(ours + ' MAPE')
    fig, ax = plt.subplots(figsize=(11, 3.8))
    x = np.arange(len(order))
    for i, name in enumerate(names):
        ax.bar(x + (i - .5) * .4, order[name + ' MAPE'], .4, label=name)
    ax.set_xticks(x, [str(f) + ('*' if r else '') for f, r in zip(order['flight'], order['random'])], rotation=90)
    ax.set_xlabel('test flight (* = also in the "random" set)')
    ax.set_ylabel('power MAPE (%)')
    ax.set_title('Per-flight power MAPE on the test flights')
    ax.grid(axis='y', alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'per_flight_mape.png'), dpi=130)
    plt.close(fig)

    # 3. example flights: best, median and worst (by our MAPE)
    picks = {'best': order.iloc[0], 'median': order.iloc[len(order) // 2], 'worst': order.iloc[-1]}
    fig, axes = plt.subplots(len(picks), 1, figsize=(11, 3 * len(picks)))
    for ax, (label, row) in zip(axes, picks.items()):
        f = int(row['flight'])
        ts = all_data[f]['time'].values[FIRST:]
        ts = ts - ts[0]
        ax.plot(ts, all_data[f]['power'].values[FIRST:], color='k', lw=1.2, label='measured')
        for name in names:
            ax.plot(ts, preds[name][f], lw=1, alpha=.85,
                    label='{} (MAPE {:.1f}%, RE {:.1f}%)'.format(name, row[name + ' MAPE'], row[name + ' RE']))
        for cp in change_points[f][1:-1]:
            ax.axvline(ts[cp], color='grey', ls=':', lw=.8)
        ax.set_title('{} flight: {} (payload {:.0f} g)'.format(label, f, all_data[f]['payload'].iloc[0] * 1000))  # stored in kg
        ax.set_ylabel('power (W)')
        ax.grid(alpha=.3)
        ax.legend(fontsize=8, loc='upper right')
    axes[-1].set_xlabel('time (s); dotted lines = regime change points used for energy RE')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'example_flights.png'), dpi=130)
    plt.close(fig)

    # same averaging as evaluate_torch.py: over the split's flight list, which repeats 269 and 278 in test
    by_flight = df.set_index('flight')
    summary = {d: {n: (by_flight.loc[ranges[d], n + ' MAPE'].mean(),
                       by_flight.loc[ranges[d], n + ' RE'].mean()) for n in names} for d in ['random', 'test']}
    print(pd.DataFrame(summary).map(lambda v: '{:.2f}/{:.2f}'.format(*v)))
    print('examples:', {k: int(v['flight']) for k, v in picks.items()})
    print('wrote', args.out)
