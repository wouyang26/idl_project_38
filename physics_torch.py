"""Liu et al. (2017) power model in PyTorch, split into its induced, profile and parasitic terms, and its fits.

Same equations and constants as `liu_model.power` (c3 = c6 = 0; air density is not used), but differentiable and
vectorized, so it can sit inside a network. Inputs are raw: vertical speed and airspeed in m/s, angle of attack in
rad, payload in kg.

    consts = fit_liu(all_data, train_flights)      # [k1, k2, c2, c4, c5], least squares on these flights only
    alpha = fit_alpha(all_data, train_flights)     # P = alpha * P_Liu with the authors' constants
"""
import numpy as np
import torch
from scipy.optimize import least_squares

import liu_model

LIU_COLS = ['vertspd', 'airspeed', 'aoa', 'payload']


def liu_terms(vertspd, airspeed, aoa, payload, consts):
    """(induced, profile, parasitic) power in W. Tensors broadcast; consts = [k1, k2, c2, c4, c5]."""
    k1, k2, c2, c4, c5 = consts
    mg = (liu_model.M + payload) * liu_model.G
    thrust = torch.sqrt((mg - c5 * (airspeed * torch.cos(aoa))**2)**2 + (c4 * airspeed**2)**2)
    induced = k1 * thrust * (vertspd / 2 + torch.sqrt((vertspd / 2)**2 + thrust / k2**2))
    profile = c2 * thrust**1.5
    parasitic = c4 * airspeed**3
    return induced, profile, parasitic


def liu_power(vertspd, airspeed, aoa, payload, consts):
    """Total Liu power in W (= liu_model.power)."""
    return sum(liu_terms(vertspd, airspeed, aoa, payload, consts))


def _stack(all_data, flights):
    """(inputs (4, N) float64 tensor, measured power (N,)) over `flights`."""
    df = np.concatenate([all_data[f][LIU_COLS + ['power']].values for f in flights])
    x = torch.as_tensor(df[:, :4].T, dtype=torch.float64)
    return x, df[:, 4]


def predict(all_data, flight, consts, alpha=1.):
    """Liu power (W) for every step of one flight."""
    x, _ = _stack(all_data, [flight])
    return alpha * liu_power(*x, torch.as_tensor(consts, dtype=torch.float64)).numpy()


def fit_liu(all_data, flights, x0=None):
    """The 5 constants, least squares on the power of `flights`, starting from the authors' values."""
    x, y = _stack(all_data, flights)
    x0 = np.asarray(liu_model.optimum_values() if x0 is None else x0, dtype=float)

    def residual(c):
        return liu_power(*x, torch.as_tensor(c)).numpy() - y

    # x_scale: the constants span 4 orders of magnitude; bounds keep the square roots real
    fit = least_squares(residual, x0, x_scale=np.abs(x0), bounds=(1e-6, np.inf))
    return fit.x.tolist()


def fit_alpha(all_data, flights, consts=None):
    """Least-squares scale alpha in P = alpha * P_Liu (authors' constants unless `consts`)."""
    x, y = _stack(all_data, flights)
    p = liu_power(*x, torch.as_tensor(liu_model.optimum_values() if consts is None else consts,
                                      dtype=torch.float64)).numpy()
    return float(p @ y / (p @ p))
