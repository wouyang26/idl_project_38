"""PyTorch port of `deep_energy_model.py` (tf.keras + keras-tcn==3.1.1).

Layer-for-layer equivalent of `create_advanced_net(..., input_type='concat')`:

    time_varying (B, lookback, 5) --> TCN | LSTM --> last step (B, F) --+
    time_invariant (B, 2) ----------------------------------------------+--> concat
        --> Linear(F+2, 10) + ReLU (power_embedding) --> Linear(10, 1) (power_reg)

Pretrained Keras checkpoints can be converted with `port_weights.py`.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# configs of the checkpoints shipped in ./models (same as evaluation.get_pre_def_configs)
PRETRAINED_CONFIGS = {
    'b-LSTM': {'net_type': 'lstm', 'dropout': 0., 'units': 64, 'n_hidden': 3, 'optimizer': 'RMSprop', 'stateful': True, 'batch_size': 1},
    's-LSTM': {'net_type': 'lstm', 'dropout': 0., 'units': 16, 'n_hidden': 3, 'optimizer': 'RMSprop', 'stateful': True, 'batch_size': 1},
    'b-TCN': {'nb_filters': 64, 'kernel_size': 2, 'nb_stacks': 1, 'n_layers': 5, 'total_epochs': 100, 'batch_size': 1},
    's-TCN': {'nb_filters': 16, 'kernel_size': 2, 'nb_stacks': 1, 'n_layers': 5, 'total_epochs': 100, 'batch_size': 1},
}


def load_inputs(**config):
    """Same defaults as `deep_energy_model.load_inputs`, overridden by `config`."""
    inputs = {
        'lookback': 20,
        'batch_size': 32,
        'nb_filters': 32,
        'kernel_size': 3,
        'nb_stacks': 1,
        'n_layers': 4,
        'total_epochs': 10,
        'tv': -1,
        'input_type': 'concat',
        'optimizer': 'Adam',
        'dropout': 0.,
        'units': 32,
        'n_hidden': 1,
        'net_type': 'tcn',
        'stateful': False,
    }
    inputs.update(config)
    inputs['dilations'] = [2**i for i in range(inputs['n_layers'])]
    return inputs


class ResidualBlock(nn.Module):
    """keras-tcn 3.1.1 `ResidualBlock` with causal padding and no normalization.

    out = relu(shape_match_conv(x) + F(x)),  F = 2 x [dilated causal conv -> relu -> spatial dropout]
    """

    def __init__(self, in_channels, nb_filters, kernel_size, dilation, dropout=0.):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation  # 'causal' padding pads the past only
        self.conv0 = nn.Conv1d(in_channels, nb_filters, kernel_size, dilation=dilation)
        self.conv1 = nn.Conv1d(nb_filters, nb_filters, kernel_size, dilation=dilation)
        self.dropout = nn.Dropout1d(dropout)  # drops whole channels, like SpatialDropout1D
        # 1x1 conv only when the channel count changes, identity otherwise
        if in_channels != nb_filters:
            self.shape_match_conv = nn.Conv1d(in_channels, nb_filters, 1)
        else:
            self.shape_match_conv = nn.Identity()

    def forward(self, x):  # x: (B, C, T)
        h = self.dropout(F.relu(self.conv0(F.pad(x, (self.left_pad, 0)))))
        h = self.dropout(F.relu(self.conv1(F.pad(h, (self.left_pad, 0)))))
        return F.relu(self.shape_match_conv(x) + h)


class TCN(nn.Module):
    """keras-tcn 3.1.1 `TCN` layer with its defaults (causal, no skip connections, relu).

    Takes and returns Keras layout (B, T, C); returns (B, nb_filters) when `return_sequences=False`.
    """

    def __init__(self, in_channels, nb_filters=64, kernel_size=2, nb_stacks=1,
                 dilations=(1, 2, 4, 8, 16, 32), dropout=0., return_sequences=False):
        super().__init__()
        self.return_sequences = return_sequences
        blocks = []
        channels = in_channels
        for _ in range(nb_stacks):
            for d in dilations:
                blocks.append(ResidualBlock(channels, nb_filters, kernel_size, d, dropout))
                channels = nb_filters
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.blocks(x.transpose(1, 2)).transpose(1, 2)
        return x if self.return_sequences else x[:, -1]


class DeepEnergyModel(nn.Module):

    def __init__(self, net_type='tcn', input_type='concat', nb_filters=32, kernel_size=3,
                 nb_stacks=1, n_layers=4, units=32, n_hidden=1, dropout=0., stateful=False, **_):
        super().__init__()
        assert net_type in ['lstm', 'tcn']
        assert input_type in ['mixed', 'concat']
        self.net_type = net_type
        self.input_type = input_type
        self.stateful = stateful

        # 'concat' feeds the 2 static features (density, payload) to the head, 'mixed' to the backbone
        n_time, n_static = (5, 2) if input_type == 'concat' else (7, 0)

        if net_type == 'tcn':
            # NOTE: the Keras version never passed `dropout` to the TCN (it was always 0)
            dilations = [2**i for i in range(n_layers)]
            self.backbone = nn.ModuleList([
                TCN(n_time if i == 0 else nb_filters, nb_filters, kernel_size, nb_stacks, dilations,
                    dropout, return_sequences=(i < n_hidden - 1))
                for i in range(n_hidden)
            ])
            n_features = nb_filters
        else:
            # Keras applies `dropout` to the inputs of every LSTM layer; nn.LSTM only between layers,
            # so add it on the input too (masks differ: Keras reuses one mask across time steps)
            self.input_dropout = nn.Dropout(dropout)
            self.backbone = nn.LSTM(n_time, units, num_layers=n_hidden, batch_first=True,
                                    dropout=dropout if n_hidden > 1 else 0.)
            n_features = units

        self.power_embedding = nn.Linear(n_features + n_static, 10)
        self.power_reg = nn.Linear(10, 1)
        self.reset_parameters()

    def forward(self, x_time, x_static=None, state=None):
        """
        x_time (B, lookback, 5|7), x_static (B, 2) or None for 'mixed',
        state: LSTM (h, c) carried over from the previous batch (stateful nets), else None.
        Returns (power prediction (B, 1), new state or None).
        """
        if self.net_type == 'tcn':
            x = x_time
            for tcn in self.backbone:
                x = tcn(x)
            features, state = x, None
        else:
            out, state = self.backbone(self.input_dropout(x_time), state)
            features = out[:, -1]

        if self.input_type == 'concat':
            features = torch.cat([features, x_static], dim=1)
        return self.power_reg(F.relu(self.power_embedding(features))), state

    @torch.no_grad()
    def reset_parameters(self):
        """Keras default initializers, so training from scratch behaves like the original."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # keras-tcn uses 'he_normal' (truncated normal, std = sqrt(2 / fan_in) / .8796)
                fan_in = m.in_channels * m.kernel_size[0]
                std = math.sqrt(2. / fan_in) / .87962566103423978
                nn.init.trunc_normal_(m.weight, 0., std, -2 * std, 2 * std)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, p in m.named_parameters():
                    if name.startswith('weight_ih'):
                        nn.init.xavier_uniform_(p)
                    elif name.startswith('weight_hh'):
                        nn.init.orthogonal_(p)
                    else:
                        nn.init.zeros_(p)
                        if name.startswith('bias_ih'):  # unit_forget_bias=True
                            p[m.hidden_size:2 * m.hidden_size] = 1.


def build_model(inputs):
    """`create_advanced_net` equivalent; `inputs` comes from `load_inputs`."""
    return DeepEnergyModel(**inputs)


def load_model(path, map_location='cpu'):
    """Load a checkpoint written by `train_torch.py` or `port_weights.py`. Returns (model, inputs, ckpt)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    model = build_model(ckpt['inputs'])
    model.load_state_dict(ckpt['state_dict'])
    return model.to(map_location), ckpt['inputs'], ckpt
