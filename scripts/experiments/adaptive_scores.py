"""Paired phase and battery-energy accounting for experiments 5 and 6."""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from adaptive_runs import ROOT, name, validate
sys.path.insert(0, str(ROOT / 'scripts'))
from phase_errors import PHASES, phases
from data_utils import find_regimes, load_dataset, process_data
from evaluate_torch import calc_energy
from fit_liu import unseen_test
from splits import get_split, load_conditions, subsample_train
from stress_test import FIRST, control_of
from torch_data import create_tensors, predict_flight
from torch_model import load_model

TERMS = ('induced', 'profile', 'parasitic')


def weights(ts):
    delta = np.diff(ts)
    if len(delta) == 0 or not np.all(np.isfinite(delta) & (delta > 0)):
        raise ValueError('scored timestamps must be strictly increasing')
    return (np.r_[0., delta] + np.r_[delta, 0.]) / 2


@torch.no_grad()
def score(model, inputs, tensors, frame, changes, meta):
    xt, xs, _ = tensors
    lo, hi = meta.pop('power_range')
    pred = predict_flight(model, xt, xs, inputs['batch_size']).astype(float) * (hi - lo) + lo
    scored = frame.iloc[FIRST:]
    t = scored.time.to_numpy()
    truth = scored.power.to_numpy()
    if pred.shape != truth.shape or not np.all(np.isfinite(pred + truth)):
        raise ValueError('invalid or misaligned predictions')
    w = weights(t)
    err = pred - truth
    labels = phases(scored.diffalt.to_numpy())
    actual = calc_energy(truth, t, changes)
    re = np.mean(abs(calc_energy(pred, t, changes) / actual - 1)) * 100
    term_data = {}
    if inputs['net_type'] == 'adaptive':
        watts, m, raw = model.term_prediction(xt, xs)
        m, raw = m.numpy().astype(float), raw.numpy().astype(float)
        delta = (m - 1) * raw
        np.testing.assert_allclose(watts.numpy().ravel(), pred, rtol=1e-5, atol=1e-3)
        np.testing.assert_allclose(delta.sum(axis=1), pred - raw.sum(axis=1), atol=1e-3)
        term_data = dict(zip(TERMS, zip(m.T, delta.T, raw.T)))
    rows = []
    for group in ['flight', *PHASES]:
        mask = np.ones(len(t), bool) if group == 'flight' else labels == group
        valid = mask & (truth > 0)
        wh = np.dot(w[mask], err[mask]) / 3600
        row = dict(meta, phase=group, samples=int(mask.sum()), valid_power=int(valid.sum()),
                   omitted_s=t[0]-frame.time.iloc[0], duration_s=w[mask].sum(),
                   mape=100*np.mean(abs(err[valid]/truth[valid])) if valid.any() else np.nan,
                   bias_w=err[mask].mean() if mask.any() else np.nan,
                   measured_wh=w[mask] @ truth[mask]/3600, predicted_wh=w[mask] @ pred[mask]/3600,
                   signed_wh=wh)
        if group == 'flight':
            row.update(regime_re=re, absolute_wh=abs(wh), underprediction_wh=max(-wh, 0))
        for term, (mult, correction, raw_w) in term_data.items():
            row[term+'_correction_wh'] = w[mask] @ correction[mask]/3600
            if mask.any():
                row.update({term+'_mult': mult[mask].mean(), term+'_correction_w': correction[mask].mean(),
                            term+'_liu_w': raw_w[mask].mean(), term+'_at_min': (mult[mask] <= 0.5).mean(),
                            term+'_at_max': (mult[mask] >= 2.).mean()})
        rows.append(row)
    for key in ['duration_s', 'measured_wh', 'predicted_wh', 'signed_wh']:
        np.testing.assert_allclose(sum(r[key] for r in rows[1:]), rows[0][key], atol=1e-9)
    return rows


def comparison_specs(split, day_only):
    nets = ['tcn', 'residual'] if day_only or split == 'windy_days' else ['tcn', 'residual', 'adaptive']
    if split == 'random':
        return [(net, split, 0, n, k) for net in nets for n in [5,10,20,35,52,87,None]
                for k in (range(3) if n else [0])]
    return [(net, s, seed, None, 0) for net, s, seed in
            itertools.product(nets, [split, control_of(split)], range(3))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('comparison', choices=['windy_days', 'wind', 'route', 'random'])
    parser.add_argument('--two-models', action='store_true', help='Experiment 5 context (plain and residual)')
    parser.add_argument('--ckpts', type=Path, default=Path('Results/ckpts'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true', help='Score only the three one-epoch, five-flight wind runs')
    args = parser.parse_args()
    specs = comparison_specs(args.comparison, args.two_models)
    if args.smoke:
        if args.comparison != 'wind':
            parser.error('--smoke requires wind')
        specs = [(net, 'wind', 0, 5, 0) for net in ['tcn', 'residual', 'adaptive']]
    paths = [args.ckpts/name(*spec)/'final.pt' for spec in specs]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        sys.exit('Missing checkpoints:\n' + '\n'.join(missing))
    data = load_dataset('data/')
    cond = load_conditions('data/')
    ids = (sorted(set(get_split(data, 'random')[0])) if args.comparison == 'random' else
           unseen_test(data, args.comparison, 'data/', cond))
    cp = find_regimes({f:data[f] for f in ids}, FIRST)
    rows = []
    fits = []
    for spec, path in zip(specs, paths):
        net, split, seed, n, k = spec
        ranges = get_split(data, split, 'data/', cond=cond)
        if n:
            ranges = subsample_train(ranges, n, k)
        validate(path, spec, ranges, epochs=1 if args.smoke else 30)
        model, cfg, ck = load_model(path)
        if set(ids).intersection(ck['ranges']['train'] + ck['ranges']['val']):
            raise ValueError(f'{path}: overlap with scored flights')
        stats = ck['data_min'], ck['data_max']
        processed = process_data(data, ranges=ranges, stats=stats, **cfg)[0]
        tensors = create_tensors(processed, ids, **cfg)
        meta = dict(arch=net+'16-leaky', activation='leaky', split=split, seed=seed,
                    train_size=len(ranges[2]), subset_seed=k)
        if net != 'tcn':
            fits.append(dict(meta, **dict(zip(['k1','k2','c2','c4','c5'],model.liu_consts.tolist()))))
        for f in ids:
            rows.extend(score(model,cfg,tensors[f],data[f],cp[f],dict(meta,flight=f,
                              power_range=(stats[0][-1],stats[1][-1]))))
    df = pd.DataFrame(rows)
    keys = ['split','seed','subset_seed','train_size','flight','phase']
    paired = df.pivot(index=keys,columns='arch',values=['signed_wh','absolute_wh','underprediction_wh','bias_w','mape','regime_re'])
    if paired['signed_wh'].isna().any().any():
        raise ValueError('missing paired flight/phase rows')
    target = 'adaptive16-leaky' if 'adaptive16-leaky' in df.arch.unique() else 'residual16-leaky'
    for other in sorted(set(df.arch)-{target}):
        for metric in ('absolute_wh','underprediction_wh','mape','regime_re','signed_wh','bias_w'):
            paired[(metric,target+'_minus_'+other)] = paired[metric,target] - paired[metric,other]
    paired.columns = ['_'.join(c) for c in paired.columns]
    by = ['arch','activation','split','train_size','phase']
    metrics = [c for c in df.select_dtypes('number') if c not in ['seed','subset_seed','flight','train_size']]
    per_seed = df.groupby(by+['seed','subset_seed'])[metrics].mean()
    per_seed['flights'] = len(ids)
    summary = per_seed.groupby(by).agg(['mean','std','count'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    args.out.mkdir(parents=True,exist_ok=True)
    for filename, table in [('per_flight',df),('paired',paired.reset_index()),
                            ('per_seed',per_seed.reset_index()),('summary',summary.reset_index()),
                            ('liu_fits',pd.DataFrame(fits))]:
        table.to_csv(args.out/(filename+'.csv'), index=False)


if __name__ == '__main__':
    main()
