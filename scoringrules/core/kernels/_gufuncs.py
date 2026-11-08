import math

import numpy as np
from numba import njit, guvectorize

from scoringrules.core.utils import lazy_gufunc_wrapper_uv, lazy_gufunc_wrapper_mv


@njit(cache=True)
def _gauss_kern_uv(x1: float, x2: float) -> float:
    """Gaussian kernel evaluated at x1 and x2."""
    out: float = math.exp(-0.5 * (x1 - x2) ** 2)
    return out


@njit(cache=True)
def _gauss_kern_mv(x1: float, x2: float) -> float:
    """Gaussian kernel evaluated at x1 and x2."""
    # sum of squares, like the array path (gauss_kern_mv in _approx.py)
    d = x1 - x2
    out: float = math.exp(-0.5 * np.sum(d * d))
    return out


@guvectorize("(),(n)->()", cache=True)
def _ks_ensemble_uv_nrg_gufunc(obs: np.ndarray, fct: np.ndarray, out: np.ndarray):
    """Standard version of the kernel score."""
    M = fct.shape[-1]

    if np.isnan(obs):
        out[0] = np.nan
        return

    # the kernel matrix is symmetric: strict upper triangle twice plus the
    # diagonal. The diagonal k(x, x) is evaluated, not set to 1, so a NaN or
    # inf member still gives NaN.
    e_1 = 0.0
    e_d = 0.0
    e_u = 0.0
    for i in range(M):
        e_1 += _gauss_kern_uv(fct[i], obs)
        e_d += _gauss_kern_uv(fct[i], fct[i])
        for j in range(i + 1, M):
            e_u += _gauss_kern_uv(fct[i], fct[j])
    e_2 = e_d + 2.0 * e_u
    e_3 = _gauss_kern_uv(obs, obs)

    out[0] = -((e_1 / M) - 0.5 * (e_2 / (M**2)) - 0.5 * e_3)


@guvectorize("(),(n)->()", cache=True)
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
        for j in range(i + 1, M):  # important to start from i + 1 and not i
            e_2 += _gauss_kern_uv(fct[j], fct[i])
    e_3 = _gauss_kern_uv(obs, obs)

    out[0] = -((e_1 / M) - e_2 / (M * (M - 1)) - 0.5 * e_3)


@guvectorize("(),(n)->()", cache=True)
def _ks_ensemble_uv_akr_gufunc(obs: np.ndarray, fct: np.ndarray, out: np.ndarray):
    """Approximate kernel representation estimator of the kernel score."""
    M = fct.shape[-1]

    if np.isnan(obs):
        out[0] = np.nan
        return

    e_1 = 0
    e_2 = 0
    for i in range(M):
        e_1 += _gauss_kern_uv(fct[i], obs)
        e_2 += _gauss_kern_uv(fct[i], fct[i - 1])
    e_3 = _gauss_kern_uv(obs, obs)

    out[0] = -(e_1 / M - 0.5 * e_2 / M - 0.5 * e_3)


@guvectorize("(),(n)->()", cache=True)
def _ks_ensemble_uv_akr_circperm_gufunc(
    obs: np.ndarray, fct: np.ndarray, out: np.ndarray
):
    """AKR estimator of the kernel score with cyclic permutation."""
    M = fct.shape[-1]

    if np.isnan(obs):
        out[0] = np.nan
        return

    e_1 = 0
    e_2 = 0
    for i in range(M):
        sigma_i = (i + M // 2) % M
        e_1 += _gauss_kern_uv(fct[i], obs)
        e_2 += _gauss_kern_uv(fct[i], fct[sigma_i])
    e_3 = _gauss_kern_uv(obs, obs)

    out[0] = -(e_1 / M - 0.5 * e_2 / M - 0.5 * e_3)


@guvectorize("(),(n),(),(n)->()", cache=True)
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


@guvectorize("(),(n),(),(n)->()", cache=True)
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


@guvectorize("(d),(m,d)->()", cache=True)
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


@guvectorize("(d),(m,d)->()", cache=True)
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

    out[0] = -(e_1 / M - e_2 / (M * (M - 1)) - 0.5 * e_3)


@guvectorize("(d),(m,d)->()", cache=True)
def _ks_ensemble_mv_akr_gufunc(obs: np.ndarray, fct: np.ndarray, out: np.ndarray):
    """Approximate kernel representation estimator of the multivariate kernel score."""
    M = fct.shape[0]

    e_1 = 0
    e_2 = 0
    for i in range(M):
        e_1 += float(_gauss_kern_mv(fct[i], obs))
        e_2 += float(_gauss_kern_mv(fct[i], fct[i - 1]))
    e_3 = float(_gauss_kern_mv(obs, obs))

    out[0] = -(e_1 / M - 0.5 * e_2 / M - 0.5 * e_3)


@guvectorize("(d),(m,d)->()", cache=True)
def _ks_ensemble_mv_akr_circperm_gufunc(
    obs: np.ndarray, fct: np.ndarray, out: np.ndarray
):
    """AKR estimator of the multivariate kernel score with cyclic permutation."""
    M = fct.shape[0]

    e_1 = 0
    e_2 = 0
    for i in range(M):
        sigma_i = (i + M // 2) % M
        e_1 += float(_gauss_kern_mv(fct[i], obs))
        e_2 += float(_gauss_kern_mv(fct[i], fct[sigma_i]))
    e_3 = float(_gauss_kern_mv(obs, obs))

    out[0] = -(e_1 / M - 0.5 * e_2 / M - 0.5 * e_3)


@guvectorize("(d),(m,d),(),(m)->()", cache=True)
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


@guvectorize("(d),(m,d),(),(m)->()", cache=True)
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
    "akr": lazy_gufunc_wrapper_uv(_ks_ensemble_uv_akr_gufunc),
    "akr_circperm": lazy_gufunc_wrapper_uv(_ks_ensemble_uv_akr_circperm_gufunc),
    "fair": lazy_gufunc_wrapper_uv(_ks_ensemble_uv_fair_gufunc),
    "nrg": lazy_gufunc_wrapper_uv(_ks_ensemble_uv_nrg_gufunc),
    "ow": lazy_gufunc_wrapper_uv(_owks_ensemble_uv_gufunc),
    "vr": lazy_gufunc_wrapper_uv(_vrks_ensemble_uv_gufunc),
}

estimator_gufuncs_mv = {
    "akr": lazy_gufunc_wrapper_mv(_ks_ensemble_mv_akr_gufunc),
    "akr_circperm": lazy_gufunc_wrapper_mv(_ks_ensemble_mv_akr_circperm_gufunc),
    "fair": lazy_gufunc_wrapper_mv(_ks_ensemble_mv_fair_gufunc),
    "nrg": lazy_gufunc_wrapper_mv(_ks_ensemble_mv_nrg_gufunc),
    "ow": lazy_gufunc_wrapper_mv(_owks_ensemble_mv_gufunc),
    "vr": lazy_gufunc_wrapper_mv(_vrks_ensemble_mv_gufunc),
}

__all__ = [
    "_ks_ensemble_uv_akr_gufunc",
    "_ks_ensemble_uv_akr_circperm_gufunc",
    "_ks_ensemble_uv_fair_gufunc",
    "_ks_ensemble_uv_nrg_gufunc",
    "_owks_ensemble_uv_gufunc",
    "_vrks_ensemble_uv_gufunc",
    "_ks_ensemble_mv_akr_gufunc",
    "_ks_ensemble_mv_akr_circperm_gufunc",
    "_ks_ensemble_mv_fair_gufunc",
    "_ks_ensemble_mv_nrg_gufunc",
    "_owks_ensemble_mv_gufunc",
    "_vrks_ensemble_mv_gufunc",
]
