import math

import numpy as np
from numba import njit, guvectorize


@njit(["float32(float32, float32)", "float64(float64, float64)"])
def _gauss_kern(x1: float, x2: float) -> float:
    """Gaussian kernel evaluated at x1 and x2."""
    out: float = math.exp(-0.5 * (x1 - x2) ** 2)
    return out


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:])",
    ],
    "(),(n)->()",
)
def _ks_ensemble_nrg_gufunc(obs: np.ndarray, fct: np.ndarray, out: np.ndarray):
    """Standard version of the kernel score."""
    obs = obs[0]
    M = fct.shape[-1]

    if np.isnan(obs):
        out[0] = np.nan
        return

    e_1 = 0
    e_2 = 0

    for x_i in fct:
        e_1 += _gauss_kern(x_i, obs)
        for x_j in fct:
            e_2 += _gauss_kern(x_i, x_j)
    e_3 = _gauss_kern(obs, obs)

    out[0] = -(e_1 / M - 0.5 * e_2 / (M**2) - 0.5 * e_3)


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:])",
    ],
    "(),(n)->()",
)
def _ks_ensemble_fair_gufunc(obs: np.ndarray, fct: np.ndarray, out: np.ndarray):
    """Fair version of the kernel score."""
    obs = obs[0]
    M = fct.shape[-1]

    if np.isnan(obs):
        out[0] = np.nan
        return

    e_1 = 0
    e_2 = 0

    for x_i in fct:
        e_1 += _gauss_kern(x_i, obs)
        for x_j in fct:
            e_2 += _gauss_kern(x_i, x_j)
    e_3 = _gauss_kern(obs, obs)

    out[0] = -(e_1 / M - 0.5 * e_2 / (M * (M - 1)) - 0.5 * e_3)


estimator_gufuncs = {
    "fair": _ks_ensemble_fair_gufunc,
    "nrg": _ks_ensemble_nrg_gufunc,
}

__all__ = [
    "_ks_ensemble_fair_gufunc",
    "_ks_ensemble_nrg_gufunc",
]
