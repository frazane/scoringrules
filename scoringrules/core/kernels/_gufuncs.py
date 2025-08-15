import math

import numpy as np
from numba import njit, guvectorize

from scoringrules.core.utils import lazy_gufunc_wrapper_uv, lazy_gufunc_wrapper_mv


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


@guvectorize("(),(n)->()")
def _ks_ensemble_uv_nrg_gufunc(obs: np.ndarray, fct: np.ndarray, out: np.ndarray):
    """Standard version of the kernel score."""
    M = fct.shape[-1]

    if np.isnan(obs):
        out[0] = np.nan
        return

    e_1 = 0
    e_2 = 0

    for i in range(M):
        e_1 += _gauss_kern_uv(fct[i], obs)
        for j in range(M):
            e_2 += _gauss_kern_uv(fct[i], fct[j])
    e_3 = _gauss_kern_uv(obs, obs)

    out[0] = -((e_1 / M) - 0.5 * (e_2 / (M**2)) - 0.5 * e_3)


@guvectorize("(),(n)->()")
def _ks_ensemble_uv_fair_gufunc(obs: np.ndarray, fct: np.ndarray, out: np.ndarray):
    """Fair version of the kernel score."""
    M = fct.shape[-1]

    if np.isnan(obs):
        out[0] = np.nan
        return

    e_1 = 0
    e_2 = 0

    for i in range(M):
        e_1 += _gauss_kern_uv(fct[i], obs)
        for j in range(i + 1, M):
            e_2 += _gauss_kern_uv(fct[i], fct[j])
    e_3 = _gauss_kern_uv(obs, obs)

    out[0] = -((e_1 / M) - e_2 / (M * (M - 1)) - 0.5 * e_3)


@guvectorize("(),(n),(),(n)->()")
def _owks_ensemble_uv_gufunc(
    obs: np.ndarray,
    fct: np.ndarray,
    ow: np.ndarray,
    fw: np.ndarray,
    out: np.ndarray,
):
    """Outcome-weighted kernel score for univariate ensembles."""
    M = fct.shape[-1]

    if np.isnan(obs):
        out[0] = np.nan
        return

    e_1 = 0.0
    e_2 = 0.0

    for i, x_i in enumerate(fct):
        e_1 += _gauss_kern_uv(x_i, obs) * fw[i] * ow
        for j, x_j in enumerate(fct):
            e_2 += _gauss_kern_uv(x_i, x_j) * fw[i] * fw[j] * ow
    e_3 = _gauss_kern_uv(obs, obs) * ow

    wbar = np.mean(fw)

    out[0] = -(e_1 / (M * wbar) - 0.5 * e_2 / (M**2 * wbar**2) - 0.5 * e_3)


@guvectorize("(),(n),(),(n)->()")
def _vrks_ensemble_uv_gufunc(
    obs: np.ndarray,
    fct: np.ndarray,
    ow: np.ndarray,
    fw: np.ndarray,
    out: np.ndarray,
):
    """Vertically re-scaled kernel score for univariate ensembles."""
    M = fct.shape[-1]

    if np.isnan(obs):
        out[0] = np.nan
        return

    e_1 = 0.0
    e_2 = 0.0

    for i, x_i in enumerate(fct):
        e_1 += _gauss_kern_uv(x_i, obs) * fw[i] * ow
        for j, x_j in enumerate(fct):
            e_2 += _gauss_kern_uv(x_i, x_j) * fw[i] * fw[j]
    e_3 = _gauss_kern_uv(obs, obs) * ow * ow

    out[0] = -((e_1 / M) - 0.5 * (e_2 / (M**2)) - 0.5 * e_3)


@guvectorize("(d),(m,d)->()")
def _ks_ensemble_mv_nrg_gufunc(obs: np.ndarray, fct: np.ndarray, out: np.ndarray):
    """Standard version of the multivariate kernel score."""
    M = fct.shape[0]

    e_1 = 0.0
    e_2 = 0.0
    for i in range(M):
        e_1 += float(_gauss_kern_mv(fct[i], obs))
        for j in range(M):
            e_2 += float(_gauss_kern_mv(fct[i], fct[j]))
    e_3 = float(_gauss_kern_mv(obs, obs))

    out[0] = -((e_1 / M) - 0.5 * (e_2 / (M**2)) - 0.5 * e_3)


@guvectorize("(d),(m,d)->()")
def _ks_ensemble_mv_fair_gufunc(obs: np.ndarray, fct: np.ndarray, out: np.ndarray):
    """Fair version of the multivariate kernel score."""
    M = fct.shape[0]

    e_1 = 0.0
    e_2 = 0.0
    for i in range(M):
        e_1 += float(_gauss_kern_mv(fct[i], obs))
        for j in range(i + 1, M):
            e_2 += float(_gauss_kern_mv(fct[i], fct[j]))
    e_3 = float(_gauss_kern_mv(obs, obs))

    out[0] = -((e_1 / M) - (e_2 / (M * (M - 1))) - 0.5 * e_3)


@guvectorize("(d),(m,d),(),(m)->()")
def _owks_ensemble_mv_gufunc(
    obs: np.ndarray,
    fct: np.ndarray,
    ow: np.ndarray,
    fw: np.ndarray,
    out: np.ndarray,
):
    """Outcome-weighted kernel score for multivariate ensembles."""
    M = fct.shape[0]

    e_1 = 0.0
    e_2 = 0.0
    for i in range(M):
        e_1 += float(_gauss_kern_mv(fct[i], obs) * fw[i] * ow)
        for j in range(M):
            e_2 += float(_gauss_kern_mv(fct[i], fct[j]) * fw[i] * fw[j] * ow)
    e_3 = float(_gauss_kern_mv(obs, obs)) * ow

    wbar = np.mean(fw)

    out[0] = -(e_1 / (M * wbar) - 0.5 * e_2 / (M**2 * wbar**2) - 0.5 * e_3)


@guvectorize("(d),(m,d),(),(m)->()")
def _vrks_ensemble_mv_gufunc(
    obs: np.ndarray,
    fct: np.ndarray,
    ow: np.ndarray,
    fw: np.ndarray,
    out: np.ndarray,
):
    """Vertically re-scaled kernel score for multivariate ensembles."""
    M = fct.shape[0]

    e_1 = 0.0
    e_2 = 0.0
    for i in range(M):
        e_1 += float(_gauss_kern_mv(fct[i], obs) * fw[i] * ow)
        for j in range(M):
            e_2 += float(_gauss_kern_mv(fct[i], fct[j]) * fw[i] * fw[j])
    e_3 = float(_gauss_kern_mv(obs, obs)) * ow * ow

    out[0] = -((e_1 / M) - 0.5 * (e_2 / (M**2)) - 0.5 * e_3)


estimator_gufuncs_uv = {
    "fair": lazy_gufunc_wrapper_uv(_ks_ensemble_uv_fair_gufunc),
    "nrg": lazy_gufunc_wrapper_uv(_ks_ensemble_uv_nrg_gufunc),
    "ow": lazy_gufunc_wrapper_uv(_owks_ensemble_uv_gufunc),
    "vr": lazy_gufunc_wrapper_uv(_vrks_ensemble_uv_gufunc),
}

estimator_gufuncs_mv = {
    "fair": lazy_gufunc_wrapper_mv(_ks_ensemble_mv_fair_gufunc),
    "nrg": lazy_gufunc_wrapper_mv(_ks_ensemble_mv_nrg_gufunc),
    "ow": lazy_gufunc_wrapper_mv(_owks_ensemble_mv_gufunc),
    "vr": lazy_gufunc_wrapper_mv(_vrks_ensemble_mv_gufunc),
}

__all__ = [
    "_ks_ensemble_uv_fair_gufunc",
    "_ks_ensemble_uv_nrg_gufunc",
    "_owks_ensemble_uv_gufunc",
    "_vrks_ensemble_uv_gufunc",
    "_ks_ensemble_mv_fair_gufunc",
    "_ks_ensemble_mv_nrg_gufunc",
    "_owks_ensemble_mv_gufunc",
    "_vrks_ensemble_mv_gufunc",
]
