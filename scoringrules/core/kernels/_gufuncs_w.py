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


@guvectorize("(),(n),(n)->()", cache=True)
def _ks_ensemble_uv_w_nrg_gufunc(
    obs: np.ndarray, fct: np.ndarray, w: np.ndarray, out: np.ndarray
):
    """Standard version of the kernel score."""
    M = fct.shape[-1]

    if np.isnan(obs):
        out[0] = np.nan
        return

    # symmetric kernel matrix: strict upper triangle twice plus the diagonal,
    # evaluated so that a NaN or inf member still gives NaN
    e_1 = 0.0
    e_d = 0.0
    e_u = 0.0
    for i in range(M):
        e_1 += _gauss_kern_uv(fct[i], obs) * w[i]
        e_d += _gauss_kern_uv(fct[i], fct[i]) * w[i] * w[i]
        for j in range(i + 1, M):
            e_u += _gauss_kern_uv(fct[i], fct[j]) * w[i] * w[j]
    e_2 = e_d + 2.0 * e_u
    e_3 = _gauss_kern_uv(obs, obs)

    out[0] = -(e_1 - 0.5 * e_2 - 0.5 * e_3)


@guvectorize("(),(n),(n)->()", cache=True)
def _ks_ensemble_uv_w_fair_gufunc(
    obs: np.ndarray, fct: np.ndarray, w: np.ndarray, out: np.ndarray
):
    """Fair version of the kernel score."""
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


@guvectorize("(),(n),(n)->()", cache=True)
def _ks_ensemble_uv_w_akr_gufunc(
    obs: np.ndarray, fct: np.ndarray, w: np.ndarray, out: np.ndarray
):
    """Approximate kernel representation estimator of the kernel score."""
    M = fct.shape[-1]

    if np.isnan(obs):
        out[0] = np.nan
        return

    e_1 = 0
    e_2 = 0
    for i in range(M):
        e_1 += _gauss_kern_uv(fct[i], obs) * w[i]
        e_2 += _gauss_kern_uv(fct[i], fct[i - 1]) * w[i]
    e_3 = _gauss_kern_uv(obs, obs)

    out[0] = -(e_1 - 0.5 * e_2 - 0.5 * e_3)


@guvectorize("(),(n),(n)->()", cache=True)
def _ks_ensemble_uv_w_akr_circperm_gufunc(
    obs: np.ndarray, fct: np.ndarray, w: np.ndarray, out: np.ndarray
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
        e_1 += _gauss_kern_uv(fct[i], obs) * w[i]
        e_2 += _gauss_kern_uv(fct[i], fct[sigma_i]) * w[i]
    e_3 = _gauss_kern_uv(obs, obs)

    out[0] = -(e_1 - 0.5 * e_2 - 0.5 * e_3)


@guvectorize("(),(n),(),(n),(n)->()", cache=True)
def _owks_ensemble_uv_w_gufunc(
    obs: np.ndarray,
    fct: np.ndarray,
    ow: np.ndarray,
    fw: np.ndarray,
    w: np.ndarray,
    out: np.ndarray,
):
    """Outcome-weighted kernel score for univariate ensembles."""

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


@guvectorize("(),(n),(),(n),(n)->()", cache=True)
def _vrks_ensemble_uv_w_gufunc(
    obs: np.ndarray,
    fct: np.ndarray,
    ow: np.ndarray,
    fw: np.ndarray,
    w: np.ndarray,
    out: np.ndarray,
):
    """Vertically re-scaled kernel score for univariate ensembles."""

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


@guvectorize("(d),(m,d),(m)->()", cache=True)
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


@guvectorize("(d),(m,d),(m)->()", cache=True)
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


@guvectorize("(d),(m,d),(m)->()", cache=True)
def _ks_ensemble_mv_w_akr_gufunc(
    obs: np.ndarray, fct: np.ndarray, w: np.ndarray, out: np.ndarray
):
    """Approximate kernel representation estimator of the multivariate kernel score."""
    M = fct.shape[0]

    e_1 = 0
    e_2 = 0
    for i in range(M):
        e_1 += float(_gauss_kern_mv(fct[i], obs) * w[i])
        e_2 += float(_gauss_kern_mv(fct[i], fct[i - 1]) * w[i])
    e_3 = float(_gauss_kern_mv(obs, obs))

    out[0] = -(e_1 - 0.5 * e_2 - 0.5 * e_3)


@guvectorize("(d),(m,d),(m)->()", cache=True)
def _ks_ensemble_mv_w_akr_circperm_gufunc(
    obs: np.ndarray, fct: np.ndarray, w: np.ndarray, out: np.ndarray
):
    """AKR estimator of the multivariate kernel score with cyclic permutation."""
    M = fct.shape[0]

    e_1 = 0
    e_2 = 0
    for i in range(M):
        sigma_i = (i + M // 2) % M
        e_1 += float(_gauss_kern_mv(fct[i], obs) * w[i])
        e_2 += float(_gauss_kern_mv(fct[i], fct[sigma_i]) * w[i])
    e_3 = float(_gauss_kern_mv(obs, obs))

    out[0] = -(e_1 - 0.5 * e_2 - 0.5 * e_3)


@guvectorize("(d),(m,d),(),(m),(m)->()", cache=True)
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


@guvectorize("(d),(m,d),(),(m),(m)->()", cache=True)
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

    e_1 = 0.0
    e_2 = 0.0
    for i in range(M):
        e_1 += float(_gauss_kern_mv(fct[i], obs) * fw[i] * ow * w[i])
        for j in range(M):
            e_2 += float(_gauss_kern_mv(fct[i], fct[j]) * fw[i] * fw[j] * w[i] * w[j])
    e_3 = float(_gauss_kern_mv(obs, obs)) * ow * ow

    out[0] = -(e_1 - 0.5 * e_2 - 0.5 * e_3)


estimator_gufuncs_uv_w = {
    "fair": lazy_gufunc_wrapper_uv(_ks_ensemble_uv_w_fair_gufunc),
    "nrg": lazy_gufunc_wrapper_uv(_ks_ensemble_uv_w_nrg_gufunc),
    "akr": lazy_gufunc_wrapper_uv(_ks_ensemble_uv_w_akr_gufunc),
    "akr_circperm": lazy_gufunc_wrapper_uv(_ks_ensemble_uv_w_akr_circperm_gufunc),
    "ow": lazy_gufunc_wrapper_uv(_owks_ensemble_uv_w_gufunc),
    "vr": lazy_gufunc_wrapper_uv(_vrks_ensemble_uv_w_gufunc),
}

estimator_gufuncs_mv_w = {
    "fair": lazy_gufunc_wrapper_mv(_ks_ensemble_mv_w_fair_gufunc),
    "nrg": lazy_gufunc_wrapper_mv(_ks_ensemble_mv_w_nrg_gufunc),
    "akr": lazy_gufunc_wrapper_mv(_ks_ensemble_mv_w_akr_gufunc),
    "akr_circperm": lazy_gufunc_wrapper_mv(_ks_ensemble_mv_w_akr_circperm_gufunc),
    "ow": lazy_gufunc_wrapper_mv(_owks_ensemble_mv_w_gufunc),
    "vr": lazy_gufunc_wrapper_mv(_vrks_ensemble_mv_w_gufunc),
}

__all__ = [
    "_ks_ensemble_uv_w_fair_gufunc",
    "_ks_ensemble_uv_w_nrg_gufunc",
    "_ks_ensemble_uv_w_akr_gufunc",
    "_ks_ensemble_uv_w_akr_circperm_gufunc",
    "_owks_ensemble_uv_w_gufunc",
    "_vrks_ensemble_uv_w_gufunc",
    "_ks_ensemble_mv_w_fair_gufunc",
    "_ks_ensemble_mv_w_nrg_gufunc",
    "_ks_ensemble_mv_w_akr_gufunc",
    "_ks_ensemble_mv_w_akr_circperm_gufunc",
    "_owks_ensemble_mv_w_gufunc",
    "_vrks_ensemble_mv_w_gufunc",
]
