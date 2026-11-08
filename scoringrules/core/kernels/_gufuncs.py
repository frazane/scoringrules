import math

import numpy as np
from numba import njit, guvectorize

from scoringrules.core.utils import lazy_gufunc_wrapper_uv, lazy_gufunc_wrapper_mv


@njit(cache=True)
def _gauss_kern_uv(x1: float, x2: float) -> float:
    """Gaussian kernel evaluated at x1 and x2."""
    out: float = math.exp(-0.5 * (x1 - x2) ** 2)
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


# Multivariate kernels: scalar sums of squares; the member pairs i < j run on a
# (D, M) copy of the members (contiguous j loop), the full double sum is the
# diagonal plus twice that. Loops are written out per gufunc, which keeps the
# first-call compile time down.


@guvectorize("(d),(m,d)->()", cache=True)
def _ks_ensemble_mv_nrg_gufunc(obs: np.ndarray, fct: np.ndarray, out: np.ndarray):
    """Standard version of the multivariate kernel score."""
    M, D = fct.shape
    e_1 = 0.0
    e_d = 0.0  # diagonal of the member pairs (1, or NaN for a non-finite member)
    for i in range(M):
        s1 = 0.0
        s0 = 0.0
        for d in range(D):
            t = float(fct[i, d] - obs[d])
            s1 += t * t
            t = float(fct[i, d] - fct[i, d])
            s0 += t * t
        e_1 += math.exp(-0.5 * s1)
        e_d += math.exp(-0.5 * s0)
    ft = np.empty((D, M), dtype=fct.dtype)
    for m in range(M):
        for d in range(D):
            ft[d, m] = fct[m, d]
    acc = np.empty(M)
    e_u = 0.0  # member pairs i < j
    for i in range(M - 1):
        for j in range(i + 1, M):
            acc[j] = 0.0
        for d in range(D):
            xi = ft[d, i]
            for j in range(i + 1, M):
                t = float(ft[d, j] - xi)
                acc[j] += t * t
        for j in range(i + 1, M):
            e_u += math.exp(-0.5 * acc[j])
    e_2 = e_d + 2 * e_u
    s3 = 0.0
    for d in range(D):
        t = float(obs[d] - obs[d])
        s3 += t * t
    e_3 = math.exp(-0.5 * s3)

    out[0] = -((e_1 / M) - 0.5 * (e_2 / (M**2)) - 0.5 * e_3)


@guvectorize("(d),(m,d)->()", cache=True)
def _ks_ensemble_mv_fair_gufunc(obs: np.ndarray, fct: np.ndarray, out: np.ndarray):
    """Fair version of the multivariate kernel score."""
    M, D = fct.shape
    e_1 = 0.0
    for i in range(M):
        s1 = 0.0
        for d in range(D):
            t = float(fct[i, d] - obs[d])
            s1 += t * t
        e_1 += math.exp(-0.5 * s1)
    ft = np.empty((D, M), dtype=fct.dtype)
    for m in range(M):
        for d in range(D):
            ft[d, m] = fct[m, d]
    acc = np.empty(M)
    e_2 = 0.0
    for i in range(M - 1):
        for j in range(i + 1, M):
            acc[j] = 0.0
        for d in range(D):
            xi = ft[d, i]
            for j in range(i + 1, M):
                t = float(ft[d, j] - xi)
                acc[j] += t * t
        for j in range(i + 1, M):
            e_2 += math.exp(-0.5 * acc[j])
    s3 = 0.0
    for d in range(D):
        t = float(obs[d] - obs[d])
        s3 += t * t
    e_3 = math.exp(-0.5 * s3)

    out[0] = -(e_1 / M - e_2 / (M * (M - 1)) - 0.5 * e_3)


@guvectorize("(d),(m,d)->()", cache=True)
def _ks_ensemble_mv_akr_gufunc(obs: np.ndarray, fct: np.ndarray, out: np.ndarray):
    """Approximate kernel representation estimator of the multivariate kernel score."""
    M, D = fct.shape
    e_1 = 0.0
    e_2 = 0.0
    for i in range(M):
        s1 = 0.0
        s2 = 0.0
        for d in range(D):
            t = float(fct[i, d] - obs[d])
            s1 += t * t
            t = float(fct[i, d] - fct[i - 1, d])
            s2 += t * t
        e_1 += math.exp(-0.5 * s1)
        e_2 += math.exp(-0.5 * s2)
    s3 = 0.0
    for d in range(D):
        t = float(obs[d] - obs[d])
        s3 += t * t
    e_3 = math.exp(-0.5 * s3)

    out[0] = -(e_1 / M - 0.5 * e_2 / M - 0.5 * e_3)


@guvectorize("(d),(m,d)->()", cache=True)
def _ks_ensemble_mv_akr_circperm_gufunc(
    obs: np.ndarray, fct: np.ndarray, out: np.ndarray
):
    """AKR estimator of the multivariate kernel score with cyclic permutation."""
    M, D = fct.shape
    e_1 = 0.0
    e_2 = 0.0
    for i in range(M):
        sigma_i = (i + M // 2) % M
        s1 = 0.0
        s2 = 0.0
        for d in range(D):
            t = float(fct[i, d] - obs[d])
            s1 += t * t
            t = float(fct[i, d] - fct[sigma_i, d])
            s2 += t * t
        e_1 += math.exp(-0.5 * s1)
        e_2 += math.exp(-0.5 * s2)
    s3 = 0.0
    for d in range(D):
        t = float(obs[d] - obs[d])
        s3 += t * t
    e_3 = math.exp(-0.5 * s3)

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
    M, D = fct.shape
    e_1 = 0.0
    e_d = 0.0
    for i in range(M):
        s1 = 0.0
        s0 = 0.0
        for d in range(D):
            t = float(fct[i, d] - obs[d])
            s1 += t * t
            t = float(fct[i, d] - fct[i, d])
            s0 += t * t
        e_1 += math.exp(-0.5 * s1) * fw[i]
        e_d += math.exp(-0.5 * s0) * fw[i] * fw[i]
    ft = np.empty((D, M), dtype=fct.dtype)
    for m in range(M):
        for d in range(D):
            ft[d, m] = fct[m, d]
    acc = np.empty(M)
    e_u = 0.0
    for i in range(M - 1):
        for j in range(i + 1, M):
            acc[j] = 0.0
        for d in range(D):
            xi = ft[d, i]
            for j in range(i + 1, M):
                t = float(ft[d, j] - xi)
                acc[j] += t * t
        fi = fw[i]
        for j in range(i + 1, M):
            e_u += math.exp(-0.5 * acc[j]) * fi * fw[j]
    e_1 = e_1 * ow
    e_2 = (e_d + 2 * e_u) * ow
    s3 = 0.0
    for d in range(D):
        t = float(obs[d] - obs[d])
        s3 += t * t
    e_3 = math.exp(-0.5 * s3) * ow

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
    M, D = fct.shape
    e_1 = 0.0
    e_d = 0.0
    for i in range(M):
        s1 = 0.0
        s0 = 0.0
        for d in range(D):
            t = float(fct[i, d] - obs[d])
            s1 += t * t
            t = float(fct[i, d] - fct[i, d])
            s0 += t * t
        e_1 += math.exp(-0.5 * s1) * fw[i]
        e_d += math.exp(-0.5 * s0) * fw[i] * fw[i]
    ft = np.empty((D, M), dtype=fct.dtype)
    for m in range(M):
        for d in range(D):
            ft[d, m] = fct[m, d]
    acc = np.empty(M)
    e_u = 0.0
    for i in range(M - 1):
        for j in range(i + 1, M):
            acc[j] = 0.0
        for d in range(D):
            xi = ft[d, i]
            for j in range(i + 1, M):
                t = float(ft[d, j] - xi)
                acc[j] += t * t
        fi = fw[i]
        for j in range(i + 1, M):
            e_u += math.exp(-0.5 * acc[j]) * fi * fw[j]
    e_1 = e_1 * ow
    e_2 = e_d + 2 * e_u
    s3 = 0.0
    for d in range(D):
        t = float(obs[d] - obs[d])
        s3 += t * t
    e_3 = math.exp(-0.5 * s3) * ow * ow

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
