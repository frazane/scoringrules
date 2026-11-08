"""Energy score gufuncs with ensemble weights ``ens_w``.

Same structure as ``_gufuncs.py`` (see there): scalar sums of squares, a ``(D, M)``
copy of the members for the spread term, loops written out per gufunc.
"""

import math

import numpy as np
from numba import guvectorize

from scoringrules.core.utils import lazy_gufunc_wrapper_mv


@guvectorize("(d),(m,d),(m)->()", cache=True)
def _energy_score_nrg_gufunc_w(obs, fct, ens_w, out):
    """Compute the Energy Score for a finite ensemble."""
    M, D = fct.shape
    e_1 = 0.0
    for i in range(M):
        s = 0.0
        for d in range(D):
            t = float(fct[i, d] - obs[d])
            s += t * t
        e_1 += math.sqrt(s) * ens_w[i]
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
        wi = ens_w[i]
        for j in range(i + 1, M):
            e_2 += math.sqrt(acc[j]) * wi * ens_w[j]
    out[0] = e_1 - 0.5 * (2 * e_2)


@guvectorize("(d),(m,d),(m)->()", cache=True)
def _energy_score_fair_gufunc_w(obs, fct, ens_w, out):
    """Compute the fair Energy Score for a finite ensemble."""
    M, D = fct.shape
    e_1 = 0.0
    for i in range(M):
        s = 0.0
        for d in range(D):
            t = float(fct[i, d] - obs[d])
            s += t * t
        e_1 += math.sqrt(s) * ens_w[i]
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
        wi = ens_w[i]
        for j in range(i + 1, M):
            e_2 += math.sqrt(acc[j]) * wi * ens_w[j]
    fair_c = 1 - np.sum(ens_w * ens_w)
    out[0] = e_1 - 0.5 * (2 * e_2) / fair_c


@guvectorize("(d),(m,d),(m)->()", cache=True)
def _energy_score_akr_gufunc_w(obs, fct, ens_w, out):
    """Compute the Energy Score for a finite ensemble using the approximate kernel representation."""
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
        e_1 += math.sqrt(s1) * ens_w[i]
        e_2 += math.sqrt(s2) * ens_w[i]
    out[0] = e_1 - 0.5 * e_2


@guvectorize("(d),(m,d),(m)->()", cache=True)
def _energy_score_akr_circperm_gufunc_w(obs, fct, ens_w, out):
    """Compute the Energy Score for a finite ensemble using the AKR with cyclic permutation."""
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
        e_1 += math.sqrt(s1) * ens_w[i]
        e_2 += math.sqrt(s2) * ens_w[i]
    out[0] = e_1 - 0.5 * e_2


@guvectorize("(d),(m,d),(),(m),(m)->()", cache=True)
def _owenergy_score_gufunc_w(obs, fct, ow, fw, ens_w, out):
    """Compute the Outcome-Weighted Energy Score for a finite ensemble."""
    M, D = fct.shape
    aw = np.empty(M)
    for i in range(M):
        aw[i] = fw[i] * ens_w[i]
    e_1 = 0.0
    for i in range(M):
        s = 0.0
        for d in range(D):
            t = float(fct[i, d] - obs[d])
            s += t * t
        e_1 += math.sqrt(s) * aw[i]
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
        ai = aw[i]
        for j in range(i + 1, M):
            e_2 += math.sqrt(acc[j]) * ai * aw[j]
    wbar = np.sum(fw * ens_w)
    out[0] = e_1 * ow / wbar - 0.5 * (2 * e_2 * ow) / (wbar**2)


@guvectorize("(d),(m,d),(),(m),(m)->()", cache=True)
def _vrenergy_score_gufunc_w(obs, fct, ow, fw, ens_w, out):
    """Compute the Vertically Re-scaled Energy Score for a finite ensemble."""
    M, D = fct.shape
    aw = np.empty(M)
    for i in range(M):
        aw[i] = fw[i] * ens_w[i]
    e_1 = 0.0
    wabs_x = 0.0
    for i in range(M):
        s = 0.0
        sx = 0.0
        for d in range(D):
            t = float(fct[i, d] - obs[d])
            s += t * t
            t = float(fct[i, d])
            sx += t * t
        e_1 += math.sqrt(s) * aw[i]
        wabs_x += math.sqrt(sx) * aw[i]
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
        ai = aw[i]
        for j in range(i + 1, M):
            e_2 += math.sqrt(acc[j]) * ai * aw[j]
    sy = 0.0
    for d in range(D):
        t = float(obs[d])
        sy += t * t
    wbar = np.sum(fw * ens_w)
    wabs_y = math.sqrt(sy) * ow
    out[0] = e_1 * ow - 0.5 * (2 * e_2) + (wabs_x - wabs_y) * (wbar - ow)


estimator_gufuncs_w = {
    "akr_circperm": lazy_gufunc_wrapper_mv(_energy_score_akr_circperm_gufunc_w),
    "akr": lazy_gufunc_wrapper_mv(_energy_score_akr_gufunc_w),
    "fair": lazy_gufunc_wrapper_mv(_energy_score_fair_gufunc_w),
    "nrg": lazy_gufunc_wrapper_mv(_energy_score_nrg_gufunc_w),
    "ownrg": lazy_gufunc_wrapper_mv(_owenergy_score_gufunc_w),
    "vrnrg": lazy_gufunc_wrapper_mv(_vrenergy_score_gufunc_w),
}

__all__ = [
    "_energy_score_akr_circperm_gufunc_w",
    "_energy_score_akr_gufunc_w",
    "_energy_score_fair_gufunc_w",
    "_energy_score_nrg_gufunc_w",
    "_owenergy_score_gufunc_w",
    "_vrenergy_score_gufunc_w",
]
