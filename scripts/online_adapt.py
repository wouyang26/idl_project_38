"""Neural-Fly-style online adaptation: refit the last layer from the power measured so far in the flight.

    python scripts/online_adapt.py                       # C checkpoints -> Results/online_adapt/
    python scripts/online_adapt.py --ckpts 'Results/ckpts/residual_*leaky_wind*/final.pt'

Everything but the last layer (power_reg: 10 weights + bias on the head features h) stays frozen, so the model is
    power = base(x) + w.h + b,    base = 0 for a TCN, Liu for the hybrid.
On each test flight, (w, b) is refit by ridge regression pulled toward the trained values,
    min ||H theta - r||^2 + lam ||theta - theta0||^2,   r = measured power - base,
using only power measured before the step being predicted. Two modes, both scored on the steps after T seconds:
    fixed    fit once on the first T seconds, then predict the rest
    online   refit at every step on everything measured so far (what a drone could run in flight)
lam is chosen on each model's own validation flights (online mode, T = 30 s, remaining-energy error), never on
test flights.

Flights: the 20 unseen windy flights of `wind` / `random_sub_wind`, and the 11 calm flights of `windy_days` /
`random_sub_windy_days` (does adapting hurt calm flights?). Scores: power MAPE and the error of the energy used over
the rest of the flight ("is there enough battery left?"), in %. Writes per_flight.csv and summary.csv.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from data_utils import load_dataset, process_data  # noqa: E402
from fit_liu import unseen_test  # noqa: E402
from splits import load_conditions  # noqa: E402
from stress_test import FIRST  # noqa: E402
from torch_data import create_tensors  # noqa: E402
from torch_model import load_model  # noqa: E402

T_ADAPT = [15, 30, 60]  # seconds of measured power before the scored part
LAMBDAS = [1e-2, 1e-1, 1, 10, 100, 1000]


@torch.no_grad()
def head_features(model, x_time, x_static):
    """Head features h (N, 10) and base = output - last layer (N,), in normalized power."""
    feats = []
    hook = model.power_reg.register_forward_hook(lambda m, inp, out: feats.append(inp[0]))
    out = torch.cat([model(x_time[i:i + 4096], x_static[i:i + 4096])[0][:, 0] for i in range(0, len(x_time), 4096)])
    hook.remove()
    h = torch.cat(feats)
    base = out - model.power_reg(h)[:, 0]
    return h.numpy().astype(np.float64), base.numpy().astype(np.float64)


def ridge(H, r, theta0, lam):
    A = H.T @ H + lam * np.eye(len(theta0))
    return np.linalg.solve(A, H.T @ r + lam * theta0)


def fixed_fit(H, r, theta0, lam, n_fit):
    """Fit on the first n_fit steps once; predicted correction for every step."""
    return H @ ridge(H[:n_fit], r[:n_fit], theta0, lam)


def online_fit(H, r, theta0, lam, n_start):
    """Step i is predicted with theta fit on steps < i (running sums, refit every step)."""
    A, c = lam * np.eye(len(theta0)), lam * theta0.copy()
    pred = H @ theta0
    for i in range(len(H)):
        if i >= n_start:
            pred[i] = H[i] @ np.linalg.solve(A, c)
        A += np.outer(H[i], H[i])
        c += H[i] * r[i]
    return pred


def errors(true, pred, t):
    """Power MAPE and the error of the energy over these steps, both in %."""
    mape = np.mean(np.abs((true - pred) / true)) * 100
    e_true, e_pred = np.trapezoid(true, t), np.trapezoid(pred, t)
    return mape, abs(e_true - e_pred) / e_true * 100


def flight_arrays(model, inputs, ckpt, all_data, flights, device):
    data, data_min, data_max, _, _, _ = process_data(all_data, stats=(ckpt['data_min'], ckpt['data_max']), **inputs)
    tensors = create_tensors(data, flights, device=device, **inputs)
    lo, span = data_min[-1], data_max[-1] - data_min[-1]
    out = {}
    for f, (xt, xs, y) in tensors.items():
        H, base = head_features(model, xt, xs)
        H = np.hstack([H, np.ones((len(H), 1))])
        t = all_data[f]['time'].values[FIRST:][:len(H)]
        out[f] = (H, base, y.numpy().astype(np.float64), t - all_data[f]['time'].values[0], lo, span)
    return out


def score(arrays, theta0, lam, mode, T):
    """Mean over flights of (MAPE, energy error) on the steps after T s, adapted (mode) and unadapted."""
    rows = []
    for f, (H, base, y, t, lo, span) in arrays.items():
        n = int(np.searchsorted(t, T))
        if n < 20 or n > len(t) - 20:
            continue
        r = y - base
        corr = fixed_fit(H, r, theta0, lam, n) if mode == 'fixed' else online_fit(H, r, theta0, lam, n)
        true = y[n:] * span + lo
        for name, c in [('none', H @ theta0), (mode, corr)]:
            mape, ee = errors(true, (base[n:] + c[n:]) * span + lo, t[n:])
            rows.append({'flight': f, 'adapt': name, 'T': T, 'mape': mape, 'energy_err': ee})
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', default='./', help='Working directory for data.')
    parser.add_argument('--ckpts', default='Results/ckpts/tcn_20_*_2_1_5_*wind*/final.pt', help='Glob of checkpoints.')
    parser.add_argument('--out', default='Results/online_adapt', help='Output directory.')
    args = parser.parse_args()

    data_dir = os.path.join(args.directory, 'data/')
    all_data = load_dataset(data_dir)
    cond = load_conditions(data_dir)
    test = {s: unseen_test(all_data, s, data_dir, cond) for s in ['wind', 'windy_days']}
    test.update({'random_sub_wind': test['wind'], 'random_sub_windy_days': test['windy_days']})

    rows = []
    for path in sorted(glob.glob(args.ckpts)):
        model, inputs, ckpt = load_model(path)
        model.eval()
        if inputs['split'] not in test or inputs.get('variance_head'):
            continue
        theta0 = np.r_[model.power_reg.weight.detach().numpy()[0], model.power_reg.bias.detach().numpy()]
        val = flight_arrays(model, inputs, ckpt, all_data, ckpt['ranges']['val'], 'cpu')
        lam = min(LAMBDAS, key=lambda l: np.mean([r['energy_err'] for r in score(val, theta0, l, 'online', 30)
                                                   if r['adapt'] != 'none']))
        arrays = flight_arrays(model, inputs, ckpt, all_data, test[inputs['split']], 'cpu')
        run = os.path.basename(os.path.dirname(path))
        size = inputs['nb_filters'] if inputs['net_type'] in ('tcn', 'residual') else inputs['units']
        arch = '{}{}{}'.format(inputs['net_type'], size, '' if inputs.get('head_act', 'relu') == 'relu' else '-leaky')
        for mode in ['fixed', 'online']:
            for T in T_ADAPT:
                for r in score(arrays, theta0, lam, mode, T):
                    if mode == 'online' and r['adapt'] == 'none':
                        continue  # same as fixed's 'none'
                    rows.append(dict(r, run=run, arch=arch, split=inputs['split'], seed=ckpt['seed'], lam=lam))
        print('{:45s} lam {:g}'.format(run, lam), flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, 'per_flight.csv'), index=False)
    per_seed = df.groupby(['arch', 'split', 'adapt', 'T', 'seed'])[['mape', 'energy_err']].mean()
    summary = per_seed.groupby(['arch', 'split', 'adapt', 'T']).agg(['mean', 'std']).round(2)
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.to_csv(os.path.join(args.out, 'summary.csv'))
    print(summary.to_string())
    print('saved', args.out)
