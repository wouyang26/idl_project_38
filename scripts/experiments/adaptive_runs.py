"""Inventory, validate and run the matched #5/#6 configurations from the repo root."""
import argparse
import itertools
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from data_utils import load_dataset
from splits import get_split, load_conditions, subsample_train
from torch_model import load_model

SETTINGS = dict(lookback=20, nb_filters=16, kernel_size=2, n_layers=5, nb_stacks=1,
                n_hidden=1, batch_size=32, optimizer='Adam', total_epochs=30,
                head_act='leaky', dropout=0., stateful=False, variance_head=False)
DAY = ['windy_days', 'random_sub_windy_days']
WIND = ['wind', 'random_sub_wind']
ROUTE = ['route', 'random_subsampled']


def matrix(stage):
    nets = ['tcn', 'residual'] if stage in ('day', 'context', 'baselines') else ['adaptive']
    if stage in ('day', 'context', 'shift', 'baselines'):
        splits = dict(day=DAY, context=WIND, shift=WIND + ROUTE, baselines=ROUTE)[stage]
        for net, split, seed in itertools.product(nets, splits, range(3)):
            yield net, split, seed, None, 0
    if stage in ('curve', 'baselines'):
        for net in nets:
            for n, k in itertools.product([5, 10, 20, 35, 52, 87], range(3)):
                yield net, 'random', 0, n, k
            yield net, 'random', 0, None, 0


def name(net, split, seed, n, k):
    return (f'{net}_20_16_2_1_5_leaky' + (f'_{split}' if split != 'random' else '')
            + (f'_n{n}_k{k}' if n else '') + (f'_s{seed}' if seed else ''))


def validate(path, spec, ranges, epochs=30):
    model, cfg, ck = load_model(path)
    net, split, seed, n, k = spec
    expected = dict(SETTINGS, total_epochs=epochs, net_type=net, split=split, train_size=n, subset_seed=k,
                    tv=-1, input_type='concat')
    mismatches = [key for key, value in expected.items() if cfg.get(key) != value]
    if mismatches or ck['seed'] != seed:
        raise ValueError(f'{path}: configuration mismatch {mismatches}, seed={ck["seed"]}')
    for key, ids in zip(['test', 'val', 'train'], ranges):
        if sorted(ids) != sorted(ck['ranges'][key]):
            raise ValueError(f'{path}: {key} flights differ')
    if net != 'tcn':
        if not torch.all(torch.isfinite(model.liu_consts) & (model.liu_consts > 0)):
            raise ValueError(f'{path}: invalid Liu fit')
        np.testing.assert_allclose(model.x_min, ck['data_min'][:5])
        np.testing.assert_allclose(model.x_max, ck['data_max'][:5])
        np.testing.assert_allclose(model.p_range, [ck['data_min'][-1], ck['data_max'][-1]])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['day', 'context', 'shift', 'curve', 'baselines'])
    p.add_argument('--manifest', type=Path, help='Write validated checkpoint paths after a complete stage')
    p.add_argument('--run', action='store_true', help='Train missing configurations (default: inventory)')
    p.add_argument('--device', default='cpu')
    p.add_argument('--ckpts', type=Path, default=Path('Results/ckpts'))
    args = p.parse_args()
    if args.ckpts.name != 'ckpts' or args.ckpts.parent.name != 'Results':
        p.error('--ckpts must end in Results/ckpts')
    data = load_dataset('data/')
    cond = load_conditions('data/')
    completed = []
    for spec in matrix(args.stage):
        net, split, seed, n, k = spec
        ranges = get_split(data, split, 'data/', cond=cond)
        if n:
            ranges = subsample_train(ranges, n, k)
        run = name(*spec)
        path = args.ckpts / run / 'final.pt'
        if path.exists():
            validate(path, spec, ranges)
            completed.append(path)
            print('reuse', path, flush=True)
            continue
        command = [sys.executable, 'train_torch.py', '--net_type', net, '--split', split,
                   '--seed', str(seed), '--device', args.device, '-o', str(args.ckpts.parent.parent)]
        for key, value in SETTINGS.items():
            if value is not False:
                command.extend(['--' + key, str(value)])
        if n:
            command.extend(['--train_size', str(n), '--subset_seed', str(k)])
        print('missing', run, flush=True)
        if args.run:
            subprocess.run(command, check=True)
            validate(path, spec, ranges)
            completed.append(path)
    if args.manifest:
        if len(completed) != len(list(matrix(args.stage))):
            sys.exit('Incomplete stage. No manifest written.')
        args.manifest.write_text(''.join(str(path) + '\n' for path in completed))
