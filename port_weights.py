"""Convert the pretrained Keras checkpoints in ./models to PyTorch state dicts.

    python port_weights.py                      # all four models -> models_torch/<name>.pt
    python port_weights.py -m s-TCN --verify    # also compare against the original Keras model

Needs tensorflow (any 2.x) to read the checkpoints. `--verify` additionally needs keras-tcn==3.1.1
and, on TF >= 2.16, tf-keras (legacy Keras 2 is selected automatically).
"""
import argparse
import os

os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')  # keras-tcn 3.1.1 needs Keras 2
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np
import tensorflow as tf
import torch

from torch_model import PRETRAINED_CONFIGS, build_model, load_inputs

SUFFIX = '/.ATTRIBUTES/VARIABLE_VALUE'


def keras_to_torch_state_dict(model, ckpt_path):
    """Map the variables of a `create_advanced_net` checkpoint onto `model` (a DeepEnergyModel)."""
    reader = tf.train.load_checkpoint(ckpt_path)
    keras_vars = {n[:-len(SUFFIX)] for n, _ in tf.train.list_variables(ckpt_path)
                  if n.startswith('layer_with_weights-') and n.endswith(SUFFIX) and '.OPTIMIZER_SLOT' not in n}
    used = set()

    def get(name):
        used.add(name)
        return torch.from_numpy(reader.get_tensor(name + SUFFIX))

    def conv(prefix):  # Keras Conv1D kernel (k, in, out) -> torch (out, in, k)
        return get(prefix + '/kernel').permute(2, 1, 0), get(prefix + '/bias')

    sd = {}
    # Keras numbers the layers that own weights: backbone layer(s) first, then the two Dense layers
    if model.net_type == 'tcn':
        for i, tcn in enumerate(model.backbone):
            for b, block in enumerate(tcn.blocks):
                src = 'layer_with_weights-{}/residual_block_{}'.format(i, b)
                dst = 'backbone.{}.blocks.{}'.format(i, b)
                names = ['conv0', 'conv1'] + (['shape_match_conv'] if isinstance(block.shape_match_conv, torch.nn.Conv1d) else [])
                for name in names:
                    keras_name = {'conv0': 'conv1D_0', 'conv1': 'conv1D_1'}.get(name, name)
                    sd[dst + '.' + name + '.weight'], sd[dst + '.' + name + '.bias'] = conv(src + '/' + keras_name)
        n_backbone = len(model.backbone)
    else:
        n_backbone = model.backbone.num_layers
        for i in range(n_backbone):
            # gate order is (input, forget, cell, output) in both frameworks; Keras has a single bias
            src = 'layer_with_weights-{}/cell'.format(i)
            sd['backbone.weight_ih_l{}'.format(i)] = get(src + '/kernel').T
            sd['backbone.weight_hh_l{}'.format(i)] = get(src + '/recurrent_kernel').T
            sd['backbone.bias_ih_l{}'.format(i)] = get(src + '/bias')
            sd['backbone.bias_hh_l{}'.format(i)] = torch.zeros_like(sd['backbone.bias_ih_l{}'.format(i)])

    for j, name in enumerate(['power_embedding', 'power_reg']):
        src = 'layer_with_weights-{}'.format(n_backbone + j)
        sd[name + '.weight'] = get(src + '/kernel').T  # Dense (in, out) -> Linear (out, in)
        sd[name + '.bias'] = get(src + '/bias')

    unused = keras_vars - used
    if unused:
        raise ValueError('Checkpoint variables not mapped (config mismatch?): {}'.format(sorted(unused)))
    return {k: v.contiguous().float() for k, v in sd.items()}


def verify(name, inputs, ckpt_path, torch_model, n_windows=64, seed=0):
    """Run the original Keras model and the port on the same random flight; return max abs difference."""
    from deep_energy_model import create_advanced_net, load_inputs as keras_load_inputs

    keras_inputs = keras_load_inputs(**PRETRAINED_CONFIGS[name])
    if hasattr(tf.keras.optimizers, 'legacy'):  # TF >= 2.11 refuses the old optimizer slots otherwise
        keras_inputs['optimizer'] = getattr(tf.keras.optimizers.legacy, keras_inputs['optimizer'])()
    keras_model = create_advanced_net((keras_inputs['lookback'], 5), **keras_inputs)
    keras_model.load_weights(ckpt_path).expect_partial()
    rng = np.random.default_rng(seed)
    x_time = rng.uniform(size=(n_windows, inputs['lookback'], 5)).astype(np.float32)
    x_static = rng.uniform(size=(n_windows, 2)).astype(np.float32)

    keras_model.reset_states()
    y_keras = keras_model.predict({'time_varying': x_time, 'time_invariant': x_static},
                                  batch_size=inputs['batch_size'], verbose=0).reshape(-1)

    from torch_data import predict_flight
    y_torch = predict_flight(torch_model, torch.from_numpy(x_time), torch.from_numpy(x_static),
                             batch_size=inputs['batch_size'], stateful=inputs['stateful'])
    return float(np.max(np.abs(y_keras - y_torch)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default=list(PRETRAINED_CONFIGS), nargs='*', choices=list(PRETRAINED_CONFIGS),
                        help='Pretrained model(s) to convert.')
    parser.add_argument('--models_dir', default='./models', help='Folder with the Keras checkpoints.')
    parser.add_argument('-o', '--output', default='./models_torch', help='Folder for the PyTorch checkpoints.')
    parser.add_argument('--verify', action='store_true', help='Compare outputs with the original Keras model.')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    for name in args.model:
        inputs = load_inputs(**PRETRAINED_CONFIGS[name])
        ckpt_path = os.path.join(args.models_dir, name, 'cp.ckpt')
        model = build_model(inputs)
        model.load_state_dict(keras_to_torch_state_dict(model, ckpt_path))  # strict: every param must be set

        out_path = os.path.join(args.output, name + '.pt')
        torch.save({'inputs': inputs, 'state_dict': model.state_dict()}, out_path)
        n_params = sum(p.numel() for p in model.parameters())
        print('{:7s} -> {} ({:,} params)'.format(name, out_path, n_params))

        if args.verify:
            print('        max |keras - torch| = {:.2e}'.format(verify(name, inputs, ckpt_path, model)))
