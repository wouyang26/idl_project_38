"""Experiment A1: Liu refit on each split's training flights, vs the authors' constants (fit on all flights).

    python fit_liu.py                       # every held-out split -> Results/liu/
    python fit_liu.py --split wind route

For each held-out split: Liu's 5 constants refit on the split's training flights, and one scale alpha
(P = alpha * P_Liu, authors' constants), both by least squares (physics_torch.py). All three are scored on the same
flights as stress_test.py: the split's test flights that neither the split nor its control trains or validates on.
Writes Results/liu/<split>.json (constants, alpha) and Results/liu/liu_refit.csv (MAPE / RE per variant).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import liu_model
from data_utils import find_regimes, load_dataset
from evaluate_torch import calc_energy
from physics_torch import fit_alpha, fit_liu, predict
from splits import CONTROLS, SPLITS, get_split, load_conditions
from stress_test import FIRST, control_of


def unseen_test(all_data, split, data_dir, cond):
    """The split's test flights that neither it nor its control trains or validates on."""
    test, val, train = get_split(all_data, split, data_dir, cond=cond)
    _, cval, ctrain = get_split(all_data, control_of(split), data_dir, cond=cond)
    return sorted(set(test) - set(val) - set(train) - set(cval) - set(ctrain))


def score(all_data, flights, cps, consts, alpha=1.):
    """Mean over flights of power MAPE and regime-energy RE (%), as in stress_test.py."""
    mape, re = [], []
    for f in flights:
        true = all_data[f]['power'].values[FIRST:]
        pred = predict(all_data, f, consts, alpha)[FIRST:]
        ts = all_data[f]['time'].values[FIRST:]
        e_true, e_pred = calc_energy(true, ts, cps[f]), calc_energy(pred, ts, cps[f])
        mape.append(np.mean(np.abs((true - pred) / true)) * 100)
        re.append(np.mean(np.abs((e_true - e_pred) / e_true)) * 100)
    return np.mean(mape), np.mean(re)


if __name__ == '__main__':
    held_out = [s for s in SPLITS[1:] if s not in CONTROLS]
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', default='./', help='Working directory for data.')
    parser.add_argument('--split', nargs='+', default=held_out, choices=held_out, help='Held-out splits.')
    parser.add_argument('--out', default='Results/liu', help='Output directory.')
    args = parser.parse_args()

    data_dir = os.path.join(args.directory, 'data/')
    all_data = load_dataset(data_dir)
    cond = load_conditions(data_dir)
    cps = find_regimes(all_data, FIRST)
    authors = liu_model.optimum_values()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for split in args.split:
        train = get_split(all_data, split, data_dir, cond=cond)[2]
        consts, alpha = fit_liu(all_data, train), fit_alpha(all_data, train)
        with open(os.path.join(args.out, split + '.json'), 'w') as f:
            json.dump({'split': split, 'n_train': len(train), 'consts': consts, 'alpha': alpha,
                       'authors_consts': authors}, f, indent=1)
        flights = unseen_test(all_data, split, data_dir, cond)
        row = {'split': split, 'n': len(flights), 'alpha': alpha}
        for name, c, a in [('authors', authors, 1.), ('refit', consts, 1.), ('alpha', authors, alpha)]:
            row[name + '_mape'], row[name + '_re'] = score(all_data, flights, cps, c, a)
        rows.append(row)
        print('{split:12s} n={n:3d}  MAPE authors {authors_mape:6.2f}  refit {refit_mape:6.2f}  alpha {alpha_mape:6.2f}'
              '   RE authors {authors_re:5.2f}  refit {refit_re:5.2f}  alpha {alpha_re:5.2f}'.format(**row))

    pd.DataFrame(rows).to_csv(os.path.join(args.out, 'liu_refit.csv'), index=False)
    print('saved', args.out)
