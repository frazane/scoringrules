import math

import numpy as np
from numba import guvectorize

from scoringrules.core.utils import lazy_gufunc_wrapper_uv, lazy_gufunc_wrapper_mv

# a Cholesky pivot below TOL_FACTOR * D * eps times its diagonal entry counts as
# zero; the same constant as ``_score.TOL_FACTOR`` (kept here so that numba's
# cache, which tracks this file only, sees changes)
TOL_FACTOR = 10.0


@lazy_gufunc_wrapper_uv
@guvectorize("(),(n),()->()", cache=True)
def _dss_uv_gufunc(obs: np.ndarray, fct: np.ndarray, bias: bool, out: np.ndarray):
    ens_mean = np.mean(fct)
    fact = len(fct) - 1.0
    if bias:
        fact += 1.0
    var = np.sum((fct - ens_mean) ** 2) / fact
    sig = np.sqrt(var)
    log_sig = 2 * np.log(sig)
    bias_precision = ((obs - ens_mean) / sig) ** 2
    out[0] = bias_precision + log_sig


@lazy_gufunc_wrapper_mv
@guvectorize("(d),(m,d),()->()", cache=True)
def _dss_mv_gufunc(obs: np.ndarray, fct: np.ndarray, bias: bool, out: np.ndarray):
    """Mahalanobis term and log-determinant through a Cholesky factorisation of the
    covariance, accumulated in float64 for every input dtype. A covariance that is not
    positive definite to the input's precision (``M <= D``, collinear or non-finite
    members, ``M == 1``) or a non-finite obs gives NaN, as on the array path
    (``ds_score_mv``)."""
    M, D = fct.shape
    if M <= D:  # the centred members span at most M - 1 dimensions
        out[0] = np.nan
        return
    mean = np.zeros(D)
    for m in range(M):
        for i in range(D):
            mean[i] += fct[m, i]
    for i in range(D):
        mean[i] /= M
    # covariance, lower triangle
    L = np.zeros((D, D))
    c = np.empty(D)
    for m in range(M):
        for i in range(D):
            c[i] = fct[m, i] - mean[i]
        for i in range(D):
            ci = c[i]
            for j in range(i + 1):
                L[i, j] += ci * c[j]
    fact = M if bias else M - 1
    for i in range(D):
        for j in range(i + 1):
            L[i, j] /= fact
    # Cholesky in place; a pivot below tol times its diagonal entry (or NaN) means
    # the covariance is singular to the input's precision
    tol = TOL_FACTOR * D * np.finfo(fct.dtype).eps
    log_det = 0.0
    for j in range(D):
        s = L[j, j]
        for k in range(j):
            s -= L[j, k] * L[j, k]
        if not s > tol * L[j, j]:
            out[0] = np.nan
            return
        log_det += math.log(s)
        ljj = math.sqrt(s)
        L[j, j] = ljj
        for i in range(j + 1, D):
            t = L[i, j]
            for k in range(j):
                t -= L[i, k] * L[j, k]
            L[i, j] = t / ljj
    # Mahalanobis term: ||z||^2 with L z = obs - mean (forward substitution); a
    # non-finite obs gives NaN, as the array path's solve does
    q = 0.0
    for i in range(D):
        t = obs[i] - mean[i]
        if not math.isfinite(t):
            out[0] = np.nan
            return
        for k in range(i):
            t -= L[i, k] * c[k]
        z = t / L[i, i]
        c[i] = z
        q += z * z
    out[0] = q + log_det
