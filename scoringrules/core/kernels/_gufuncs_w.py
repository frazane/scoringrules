import math

import numpy as np
from numba import njit, guvectorize


@njit(["float32(float32, float32)", "float64(float64, float64)"])
def _gauss_kern_uv(x1: float, x2: float) -> float:
    """Gaussian kernel evaluated at x1 and x2."""
    out: float = math.exp(-0.5 * (x1 - x2) ** 2)
    return out


@njit(["float32(float32[:], float32[:])", "float64(float64[:], float64[:])"])
def _gauss_kern_mv(x1: float, x2: float) -> float:
    """Gaussian kernel evaluated at x1 and x2."""
    out: float = math.exp(-0.5 * np.linalg.norm(x1 - x2) ** 2)
    return out


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:], float64[:])",
    ],
    "(),(n),(n)->()",
)
def _ks_ensemble_uv_w_nrg_gufunc(
    obs: np.ndarray, fct: np.ndarray, w: np.ndarray, out: np.ndarray
):
    """Standard version of the kernel score."""
    obs = obs[0]
    M = fct.shape[-1]

    if np.isnan(obs):
        out[0] = np.nan
        return

    e_1 = 0
    e_2 = 0

    for i in range(M):
        e_1 += _gauss_kern_uv(fct[i], obs) * w[i]
        for j in range(M):
            e_2 += _gauss_kern_uv(fct[i], fct[j]) * w[i] * w[j]
    e_3 = _gauss_kern_uv(obs, obs)

    out[0] = -(e_1 - 0.5 * e_2 - 0.5 * e_3)


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:], float64[:])",
    ],
    "(),(n),(n)->()",
)
def _ks_ensemble_uv_w_fair_gufunc(
    obs: np.ndarray, fct: np.ndarray, w: np.ndarray, out: np.ndarray
):
    """Fair version of the kernel score."""
    obs = obs[0]
    M = fct.shape[-1]

    if np.isnan(obs):
        out[0] = np.nan
        return

    e_1 = 0
    e_2 = 0

    for i in range(M):
        e_1 += _gauss_kern_uv(fct[i], obs) * w[i]
        for j in range(i + 1, M):
            e_2 += _gauss_kern_uv(fct[i], fct[j]) * w[i] * w[j]
    e_3 = _gauss_kern_uv(obs, obs)

    out[0] = -(e_1 - e_2 / (1 - np.sum(w * w)) - 0.5 * e_3)


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:], float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:])",
    ],
    "(),(n),(),(n),(n)->()",
)
def _owks_ensemble_uv_w_gufunc(
    obs: np.ndarray,
    fct: np.ndarray,
    ow: np.ndarray,
    fw: np.ndarray,
    w: np.ndarray,
    out: np.ndarray,
):
    """Outcome-weighted kernel score for univariate ensembles."""
    obs = obs[0]
    ow = ow[0]

    if np.isnan(obs):
        out[0] = np.nan
        return

    e_1 = 0.0
    e_2 = 0.0

    for i, x_i in enumerate(fct):
        e_1 += _gauss_kern_uv(x_i, obs) * fw[i] * ow * w[i]
        for j, x_j in enumerate(fct):
            e_2 += _gauss_kern_uv(x_i, x_j) * fw[i] * fw[j] * ow * w[i] * w[j]
    e_3 = _gauss_kern_uv(obs, obs) * ow

    wbar = np.sum(fw * w)

    out[0] = -(e_1 / wbar - 0.5 * e_2 / (wbar**2) - 0.5 * e_3)


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:], float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:])",
    ],
    "(),(n),(),(n),(n)->()",
)
def _vrks_ensemble_uv_w_gufunc(
    obs: np.ndarray,
    fct: np.ndarray,
    ow: np.ndarray,
    fw: np.ndarray,
    w: np.ndarray,
    out: np.ndarray,
):
    """Vertically re-scaled kernel score for univariate ensembles."""
    obs = obs[0]
    ow = ow[0]

    if np.isnan(obs):
        out[0] = np.nan
        return

    e_1 = 0.0
    e_2 = 0.0

    for i, x_i in enumerate(fct):
        e_1 += _gauss_kern_uv(x_i, obs) * fw[i] * ow * w[i]
        for j, x_j in enumerate(fct):
            e_2 += _gauss_kern_uv(x_i, x_j) * fw[i] * fw[j] * w[i] * w[j]
    e_3 = _gauss_kern_uv(obs, obs) * ow * ow

    out[0] = -(e_1 - 0.5 * e_2 - 0.5 * e_3)


@guvectorize(
    [
        "void(float32[:], float32[:,:], float32[:], float32[:])",
        "void(float64[:], float64[:,:], float64[:], float64[:])",
    ],
    "(d),(m,d),(m)->()",
)
def _ks_ensemble_mv_w_nrg_gufunc(
    obs: np.ndarray, fct: np.ndarray, w: np.ndarray, out: np.ndarray
):
    """Standard version of the multivariate kernel score."""
    M = fct.shape[0]

    e_1 = 0.0
    e_2 = 0.0
    for i in range(M):
        e_1 += float(_gauss_kern_mv(fct[i], obs)) * w[i]
        for j in range(M):
            e_2 += float(_gauss_kern_mv(fct[i], fct[j])) * w[i] * w[j]
    e_3 = float(_gauss_kern_mv(obs, obs))

    out[0] = -(e_1 - 0.5 * e_2 - 0.5 * e_3)


@guvectorize(
    [
        "void(float32[:], float32[:,:], float32[:], float32[:])",
        "void(float64[:], float64[:,:], float64[:], float64[:])",
    ],
    "(d),(m,d),(m)->()",
)
def _ks_ensemble_mv_w_fair_gufunc(
    obs: np.ndarray, fct: np.ndarray, w: np.ndarray, out: np.ndarray
):
    """Fair version of the multivariate kernel score."""
    M = fct.shape[0]

    e_1 = 0.0
    e_2 = 0.0
    for i in range(M):
        e_1 += float(_gauss_kern_mv(fct[i], obs) * w[i])
        for j in range(i + 1, M):
            e_2 += float(_gauss_kern_mv(fct[i], fct[j]) * w[i] * w[j])
    e_3 = float(_gauss_kern_mv(obs, obs))

    out[0] = -(e_1 - e_2 / (1 - np.sum(w * w)) - 0.5 * e_3)


@guvectorize(
    [
        "void(float32[:], float32[:,:], float32[:], float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:,:], float64[:], float64[:], float64[:], float64[:])",
    ],
    "(d),(m,d),(),(m),(m)->()",
)
def _owks_ensemble_mv_w_gufunc(
    obs: np.ndarray,
    fct: np.ndarray,
    ow: np.ndarray,
    fw: np.ndarray,
    w: np.ndarray,
    out: np.ndarray,
):
    """Outcome-weighted kernel score for multivariate ensembles."""
    M = fct.shape[0]
    ow = ow[0]

    e_1 = 0.0
    e_2 = 0.0
    for i in range(M):
        e_1 += float(_gauss_kern_mv(fct[i], obs) * fw[i] * ow * w[i])
        for j in range(M):
            e_2 += float(
                _gauss_kern_mv(fct[i], fct[j]) * fw[i] * fw[j] * ow * w[i] * w[j]
            )
    e_3 = float(_gauss_kern_mv(obs, obs)) * ow

    wbar = np.sum(fw * w)

    out[0] = -(e_1 / wbar - 0.5 * e_2 / (wbar**2) - 0.5 * e_3)


@guvectorize(
    [
        "void(float32[:], float32[:,:], float32[:], float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:,:], float64[:], float64[:], float64[:], float64[:])",
    ],
    "(d),(m,d),(),(m),(m)->()",
)
def _vrks_ensemble_mv_w_gufunc(
    obs: np.ndarray,
    fct: np.ndarray,
    ow: np.ndarray,
    fw: np.ndarray,
    w: np.ndarray,
    out: np.ndarray,
):
    """Vertically re-scaled kernel score for multivariate ensembles."""
    M = fct.shape[0]
    ow = ow[0]

    e_1 = 0.0
    e_2 = 0.0
    for i in range(M):
        e_1 += float(_gauss_kern_mv(fct[i], obs) * fw[i] * ow * w[i])
        for j in range(M):
            e_2 += float(_gauss_kern_mv(fct[i], fct[j]) * fw[i] * fw[j] * w[i] * w[j])
    e_3 = float(_gauss_kern_mv(obs, obs)) * ow * ow

    out[0] = -(e_1 - 0.5 * e_2 - 0.5 * e_3)


estimator_gufuncs_uv_w = {
    "fair": _ks_ensemble_uv_w_fair_gufunc,
    "nrg": _ks_ensemble_uv_w_nrg_gufunc,
    "ow": _owks_ensemble_uv_w_gufunc,
    "vr": _vrks_ensemble_uv_w_gufunc,
}

estimator_gufuncs_mv_w = {
    "fair": _ks_ensemble_mv_w_fair_gufunc,
    "nrg": _ks_ensemble_mv_w_nrg_gufunc,
    "ow": _owks_ensemble_mv_w_gufunc,
    "vr": _vrks_ensemble_mv_w_gufunc,
}

__all__ = [
    "_ks_ensemble_uv_w_fair_gufunc",
    "_ks_ensemble_uv_w_nrg_gufunc",
    "_owks_ensemble_uv_w_gufunc",
    "_vrks_ensemble_uv_w_gufunc",
    "_ks_ensemble_mv_w_fair_gufunc",
    "_ks_ensemble_mv_w_nrg_gufunc",
    "_owks_ensemble_mv_w_gufunc",
    "_vrks_ensemble_mv_w_gufunc",
]
