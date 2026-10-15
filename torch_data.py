"""PyTorch replacements for the tf.data parts of `data_utils.py` (`create_tensors`, `model.predict`)."""
import numpy as np
import torch

VARYING_COLS = [0, 1, 2, 3, 4]  # airspeed_x, airspeed_y, vertspd, aoa, airspeed
FIXED_COLS = [5, 6]             # density, payload


def create_tensors(data, flight_range, input_type='concat', device='cpu', **kwargs):
    """Torch counterpart of `data_utils.create_tensors`.

    data: first output of `data_utils.process_data`, {flight: (x (N, lookback, 7), y (N,))}
    Returns {flight: (x_time, x_static, y)} float32 tensors on `device`.
    """
    assert input_type in ['mixed', 'concat']
    cols = VARYING_COLS if input_type == 'concat' else VARYING_COLS + FIXED_COLS

    result = {}
    for flight in flight_range:
        x, y = data[flight]
        if len(x.shape) != 3:  # flight shorter than the lookback window
            continue
        # density and payload are constant over a flight, so any time step works
        tensors = (x[:, :, cols], x[:, -1, FIXED_COLS], y)
        result[flight] = tuple(torch.as_tensor(np.asarray(t, dtype=np.float32), device=device) for t in tensors)
    return result


def iterate_batches(tensors, batch_size, drop_remainder=False):
    """In-order mini-batches of one flight's tensors (order matters for stateful LSTMs)."""
    n = len(tensors[0])
    stop = n - n % batch_size if drop_remainder else n
    for i in range(0, stop, batch_size):
        yield tuple(t[i:i + batch_size] for t in tensors)


@torch.no_grad()
def predict_flight(model, x_time, x_static, batch_size=1, stateful=False):
    """Torch counterpart of `model.reset_states(); model.predict(flight)`. Returns normalized power (N,)."""
    model.eval()
    # a stateful Keras net carries the state of sample i of a batch to sample i of the next one,
    # which only works with equal-sized batches
    drop_remainder = stateful and batch_size > 1
    state = None
    preds = []
    for xt, xs in iterate_batches((x_time, x_static), batch_size, drop_remainder):
        pred, new_state = model(xt, xs, state)
        if stateful:
            state = new_state
        preds.append(pred.squeeze(-1))
    return torch.cat(preds).cpu().numpy()


def regression_metrics(y_pred, y_true):
    """Keras' MSE, RootMeanSquaredError and MeanAbsolutePercentageError (epsilon 1e-7)."""
    err = y_pred - y_true
    mse = float(torch.mean(err**2))
    mape = float(100. * torch.mean(torch.abs(err) / torch.clamp(torch.abs(y_true), min=1e-7)))
    return mse, mse**.5, mape
