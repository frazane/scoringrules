"""Energy score gufuncs.

Distances are scalar sums of squares (no temporaries). The spread term over member
pairs ``i < j`` runs on a ``(D, M)`` copy of the members, so that for one member
``i`` the loop over ``j`` is contiguous. The loops are written out in every gufunc
rather than shared through njit helpers, which keeps the first-call compile time
down.
"""

import math

import numpy as np
from numba import guvectorize

from scoringrules.core.utils import lazy_gufunc_wrapper_mv


@guvectorize("(d),(m,d)->()", cache=True)
def _energy_score_nrg_gufunc(obs, fct, out):
    """Compute the Energy Score for a finite ensemble."""
    M, D = fct.shape
    e_1 = 0.0
    for i in range(M):
        s = 0.0
        for d in range(D):
            t = float(fct[i, d] - obs[d])
            s += t * t
        e_1 += math.sqrt(s)
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
            e_2 += math.sqrt(acc[j])
    out[0] = e_1 / M - 0.5 / (M**2) * (2 * e_2)


@guvectorize("(d),(m,d)->()", cache=True)
def _energy_score_fair_gufunc(obs, fct, out):
    """Compute the fair Energy Score for a finite ensemble."""
    M, D = fct.shape
    e_1 = 0.0
    for i in range(M):
        s = 0.0
        for d in range(D):
            t = float(fct[i, d] - obs[d])
            s += t * t
        e_1 += math.sqrt(s)
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
            e_2 += math.sqrt(acc[j])
    out[0] = e_1 / M - 0.5 / (M * (M - 1)) * (2 * e_2)


@guvectorize("(d),(m,d)->()", cache=True)
def _energy_score_akr_gufunc(obs, fct, out):
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
        e_1 += math.sqrt(s1)
        e_2 += math.sqrt(s2)
    out[0] = e_1 / M - 0.5 * 1 / M * e_2


@guvectorize("(d),(m,d)->()", cache=True)
def _energy_score_akr_circperm_gufunc(obs, fct, out):
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
        e_1 += math.sqrt(s1)
        e_2 += math.sqrt(s2)
    out[0] = e_1 / M - 0.5 * 1 / M * e_2


@guvectorize("(d),(m,d),(),(m)->()", cache=True)
def _owenergy_score_gufunc(obs, fct, ow, fw, out):
    """Compute the Outcome-Weighted Energy Score for a finite ensemble."""
    M, D = fct.shape
    e_1 = 0.0
    for i in range(M):
        s = 0.0
        for d in range(D):
            t = float(fct[i, d] - obs[d])
            s += t * t
        e_1 += math.sqrt(s) * fw[i]
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
        fi = fw[i]
        for j in range(i + 1, M):
            e_2 += math.sqrt(acc[j]) * fi * fw[j]
    wbar = np.mean(fw)
    out[0] = e_1 * ow / (M * wbar) - 0.5 * (2 * e_2 * ow) / (M**2 * wbar**2)


@guvectorize("(d),(m,d),(),(m)->()", cache=True)
def _vrenergy_score_gufunc(obs, fct, ow, fw, out):
    """Compute the Vertically Re-scaled Energy Score for a finite ensemble."""
    M, D = fct.shape
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
        e_1 += math.sqrt(s) * fw[i]
        wabs_x += math.sqrt(sx) * fw[i]
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
        fi = fw[i]
        for j in range(i + 1, M):
            e_2 += math.sqrt(acc[j]) * fi * fw[j]
    sy = 0.0
    for d in range(D):
        t = float(obs[d])
        sy += t * t
    wabs_x = wabs_x / M
    wbar = np.mean(fw)
    wabs_y = math.sqrt(sy) * ow
    out[0] = e_1 * ow / M - 0.5 * (2 * e_2) / (M**2) + (wabs_x - wabs_y) * (wbar - ow)


estimator_gufuncs = {
    "akr_circperm": lazy_gufunc_wrapper_mv(_energy_score_akr_circperm_gufunc),
    "akr": lazy_gufunc_wrapper_mv(_energy_score_akr_gufunc),
    "fair": lazy_gufunc_wrapper_mv(_energy_score_fair_gufunc),
    "nrg": lazy_gufunc_wrapper_mv(_energy_score_nrg_gufunc),
    "ownrg": lazy_gufunc_wrapper_mv(_owenergy_score_gufunc),
    "vrnrg": lazy_gufunc_wrapper_mv(_vrenergy_score_gufunc),
}

__all__ = [
    "_energy_score_akr_circperm_gufunc",
    "_energy_score_akr_gufunc",
    "_energy_score_fair_gufunc",
    "_energy_score_nrg_gufunc",
    "_owenergy_score_gufunc",
    "_vrenergy_score_gufunc",
]
