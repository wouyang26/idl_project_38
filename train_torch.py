"""PyTorch version of `train.py`. Same arguments, plus --n_hidden, --save_every, --device and --seed.

    python train_torch.py --net_type tcn --nb_filters 16 --kernel_size 2 --n_layers 5 --total_epochs 100
    python train_torch.py --net_type lstm --units 16 --n_hidden 3 --optimizer RMSprop --stateful --batch_size 1

Differences from train.py:
  - one "epoch" here is one pass over all training flights (train.py counted every flight as an epoch);
    validation, logging and checkpoints happen per epoch
  - the last partial batch of a flight is kept (train.py meant to, but `drop_remainer=False` was a typo),
    except for stateful nets, which need equal-sized batches
"""
import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data_utils import get_dataset, load_dataset, process_data
from torch_data import create_tensors, iterate_batches, predict_flight, regression_metrics
from torch_model import build_model, load_inputs

# Keras default hyperparameters for the optimizers train.py accepts by name
OPTIMIZERS = {
    'Adam': lambda p: torch.optim.Adam(p, lr=1e-3, eps=1e-7),
    'RMSprop': lambda p: torch.optim.RMSprop(p, lr=1e-3, alpha=.9, eps=1e-7),
    'SGD': lambda p: torch.optim.SGD(p, lr=1e-2),
}


def train_flight(model, optimizer, tensors, batch_size, stateful):
    """One pass over one flight (train.py's `model.fit(train_tensors[flight])` + `reset_states()`)."""
    model.train()
    state = None
    preds, targets = [], []
    for x_time, x_static, y in iterate_batches(tensors, batch_size, drop_remainder=stateful):
        pred, new_state = model(x_time, x_static, state)
        pred = pred.squeeze(-1)
        loss = F.mse_loss(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if stateful:  # carry the state to the next batch, but don't backpropagate through it
            state = tuple(s.detach() for s in new_state)
        preds.append(pred.detach())
        targets.append(y)
    if not preds:
        return None
    return regression_metrics(torch.cat(preds), torch.cat(targets))


def evaluate(model, flights, batch_size, stateful):
    """Average over flights of (loss, rmse, mape), skipping unstable flights like train.py's callback."""
    results = []
    for x_time, x_static, y in flights.values():
        y_pred = torch.as_tensor(predict_flight(model, x_time, x_static, batch_size, stateful), device=y.device)
        metrics = regression_metrics(y_pred, y[:len(y_pred)])
        if metrics[2] <= 10000:  # very high value; for instability
            results.append(metrics)
    return np.mean(results, axis=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', default='./', help='Working directory for data.')
    parser.add_argument('-o', '--output', default='./', help='Output directory for training.')
    parser.add_argument('-D', '--data', action='store_true', help='Whether to download the data or not.')
    parser.add_argument('--lookback', default=20, type=int, help='Size of lookback window')
    parser.add_argument('--batch_size', default=32, type=int, help='Size of the mini-batches.')
    parser.add_argument('--nb_filters', default=32, type=int, help='(For TCN) The number of filters to use in the convolutional layers.')
    parser.add_argument('--kernel_size', default=3, type=int, help='(For TCN) The size of the kernel to use in each convolutional layer.')
    parser.add_argument('--nb_stacks', default=1, type=int, help='(For TCN) The number of stacks of residual blocks to use.')
    parser.add_argument('--n_layers', default=4, type=int, help='(For TCN) Number of residual blocks per stack (dilations 1, 2, ..., 2**(n_layers-1)).')
    parser.add_argument('--n_hidden', default=1, type=int, help='Number of stacked LSTM / TCN layers (the paper LSTMs use 3).')
    parser.add_argument('--total_epochs', default=10, type=int, help='Total number passes over the entire training data set.')
    parser.add_argument('--optimizer', default='Adam', choices=list(OPTIMIZERS), help='Optimizer used by the neural network.')
    parser.add_argument('--dropout', default=0., type=float, help='Dropout used by the neural network.')
    parser.add_argument('--units', default=32, type=int, help='(For LSTM) The number of units in each layer.')
    parser.add_argument('--net_type', default='tcn', choices=['lstm', 'tcn'], help='The type of net to train.')
    parser.add_argument('--stateful', action='store_true', help='Carry the LSTM state across batches of a flight.')
    parser.add_argument('--save_every', default=10, type=int, help='Save a checkpoint every this many epochs.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', default=0, type=int, help='Seed for the weight initialization / dropout.')
    args = parser.parse_args()

    directory = os.path.join(args.directory, '')
    if not os.path.exists(directory):
        raise ValueError('Invalid data path')
    if args.data:
        get_dataset(directory)
    torch.manual_seed(args.seed)

    all_data = load_dataset(directory + 'data/')

    config = {k: v for k, v in vars(args).items() if k not in ['directory', 'output', 'data', 'save_every', 'device', 'seed']}
    inputs = load_inputs(**config)
    stateful = inputs['net_type'] == 'lstm' and inputs['stateful']

    data, data_min, data_max, test_range, train_range, val_range = process_data(all_data, **inputs)
    train_tensors = create_tensors(data, train_range, device=args.device, **inputs)
    val_tensors = create_tensors(data, val_range, device=args.device, **inputs)
    train_range = np.array([f for f in train_range if f in train_tensors])

    if inputs['net_type'] == 'lstm':
        save_str = '{net_type}_{lookback}_{dropout}_{units}_{n_hidden}_{optimizer}'.format(**inputs)
    else:
        save_str = '{net_type}_{lookback}_{nb_filters}_{kernel_size}_{nb_stacks}_{n_layers}'.format(**inputs)
    output_dir = os.path.join(args.output, 'Results')
    ckpt_dir = Path(output_dir, 'ckpts', save_str)
    csv_dir = Path(output_dir, 'CsvResults', save_str)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(inputs).to(args.device)
    optimizer = OPTIMIZERS[inputs['optimizer']](model.parameters())
    print(model)
    print('{:,} parameters, training on {} flights, validating on {}'.format(
        sum(p.numel() for p in model.parameters()), len(train_tensors), len(val_tensors)))

    def save(path):
        # everything needed to rebuild the model and undo the power normalization
        torch.save({'inputs': inputs, 'state_dict': model.state_dict(),
                    'data_min': data_min.tolist(), 'data_max': data_max.tolist()}, path)

    save(ckpt_dir / 'epoch_0000.pt')
    history = []
    for epoch in range(1, inputs['total_epochs'] + 1):
        # same flight order as train.py (re-seeded in-place shuffle every epoch)
        np.random.seed(42)
        np.random.shuffle(train_range)

        train_metrics = [m for m in (train_flight(model, optimizer, train_tensors[f], inputs['batch_size'], stateful)
                                     for f in train_range) if m is not None]
        train_loss, train_rmse, train_mape = np.mean(train_metrics, axis=0)
        val_loss, val_rmse, val_mape = evaluate(model, val_tensors, inputs['batch_size'], stateful)

        history.append([epoch, train_loss, train_rmse, train_mape, val_loss, val_rmse, val_mape])
        print('epoch {:4d} | train loss {:.6f} rmse {:.4f} mape {:8.2f} | val loss {:.6f} rmse {:.4f} mape {:8.2f}'.format(*history[-1]))
        if epoch % args.save_every == 0:
            save(ckpt_dir / 'epoch_{:04d}.pt'.format(epoch))

    save(ckpt_dir / 'final.pt')
    with open(csv_dir / 'csv_result.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Training Loss', 'Training RMSE', 'Training MAPE',
                         'Validation Loss', 'Validation RMSE', 'Validation MAPE'])
        writer.writerows(history)
    print('saved', ckpt_dir / 'final.pt', 'and', csv_dir / 'csv_result.csv')
