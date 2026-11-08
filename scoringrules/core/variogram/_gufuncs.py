"""Variogram score gufuncs.

All of them work on unordered pairs of variables ``i <= j``: ``|x_i - x_j| ** p`` is
symmetric, so the pair ``(j, i)`` enters through the weight ``w_ij + w_ji``. The
diagonal is kept (its terms are zero for finite members) so that NaN and inf
propagate as with a full ``(i, j)`` loop. ``p`` equal to 0.5, 1 or 2 uses sqrt, abs
or a square instead of ``pow``, chosen once per call.

The members are copied to a ``(D, M)`` array, so for one pair the member loop is
contiguous. The fair, ``ow`` and ``vr`` spread term is
``sum_{k<m} a_k a_m (r_k - r_m) ** 2 = S0 S2 - S1 ** 2`` with ``S0 = sum a_k``,
``S1 = sum a_k t_k``, ``S2 = sum a_k t_k ** 2`` and ``t_k = r_k - r_c`` shifted by the
member ``c`` of largest weight, which bounds the cancellation. This is O(M D^2)
instead of O(M^2 D^2).

The loops are written out in every gufunc rather than shared through njit helpers,
which keeps the first-call compile time down.
"""

import math

import numpy as np
from numba import guvectorize

from scoringrules.core.utils import lazy_gufunc_wrapper_mv


@guvectorize("(d),(m,d),(d,d),()->()", cache=True)
def _variogram_score_nrg_gufunc(obs, fct, w, p, out):
    M, D = fct.shape
    # 1: sqrt, 2: abs, 3: square, 0: pow
    mode = 1 if p == 0.5 else 2 if p == 1.0 else 3 if p == 2.0 else 0
    c0 = abs(0.0) ** p  # the diagonal value for finite members
    cheap_diag = p != 0.0  # NaN ** 0 is 1, so p == 0 evaluates the diagonal
    ft = np.empty((D, M), dtype=fct.dtype)
    for m in range(M):
        for d in range(D):
            ft[d, m] = fct[m, d]
    g = np.empty(M)
    vf = np.empty((D, D))
    vo = np.empty((D, D))
    for i in range(D):
        for j in range(i, D):
            if i == j and cheap_diag:
                for m in range(M):
                    g[m] = c0 + float(ft[i, m] - ft[i, m])
            elif mode == 1:
                for m in range(M):
                    g[m] = math.sqrt(abs(float(ft[i, m] - ft[j, m])))
            elif mode == 2:
                for m in range(M):
                    g[m] = abs(float(ft[i, m] - ft[j, m]))
            elif mode == 3:
                for m in range(M):
                    t = float(ft[i, m] - ft[j, m])
                    g[m] = t * t
            else:
                for m in range(M):
                    g[m] = abs(float(ft[i, m] - ft[j, m])) ** p
            s = 0.0
            for m in range(M):
                s += g[m]
            vf[i, j] = vf[j, i] = s / M
            a = abs(float(obs[i] - obs[j]))
            if mode == 1:
                vo[i, j] = vo[j, i] = math.sqrt(a)
            elif mode == 2:
                vo[i, j] = vo[j, i] = a
            elif mode == 3:
                vo[i, j] = vo[j, i] = a * a
            else:
                vo[i, j] = vo[j, i] = a**p
    e = 0.0
    for i in range(D):
        for j in range(D):
            e += w[i, j] * (vo[i, j] - vf[i, j]) ** 2
    out[0] = e


@guvectorize("(d),(m,d),(d,d),()->()", cache=True)
def _variogram_score_fair_gufunc(obs, fct, w, p, out):
    M, D = fct.shape
    mode = 1 if p == 0.5 else 2 if p == 1.0 else 3 if p == 2.0 else 0
    c0 = abs(0.0) ** p
    cheap_diag = p != 0.0
    ft = np.empty((D, M), dtype=fct.dtype)
    for m in range(M):
        for d in range(D):
            ft[d, m] = fct[m, d]
    g = np.empty(M)
    e_1 = 0.0
    e_2 = 0.0
    for i in range(D):
        for j in range(i, D):
            if i == j and cheap_diag:
                for m in range(M):
                    g[m] = c0 + float(ft[i, m] - ft[i, m])
            elif mode == 1:
                for m in range(M):
                    g[m] = math.sqrt(abs(float(ft[i, m] - ft[j, m])))
            elif mode == 2:
                for m in range(M):
                    g[m] = abs(float(ft[i, m] - ft[j, m]))
            elif mode == 3:
                for m in range(M):
                    t = float(ft[i, m] - ft[j, m])
                    g[m] = t * t
            else:
                for m in range(M):
                    g[m] = abs(float(ft[i, m] - ft[j, m])) ** p
            a = abs(float(obs[i] - obs[j]))
            if mode == 1:
                ro = math.sqrt(a)
            elif mode == 2:
                ro = a
            elif mode == 3:
                ro = a * a
            else:
                ro = a**p
            rc = g[0]
            s1 = 0.0
            s2 = 0.0
            e = 0.0
            for m in range(M):
                t = g[m] - rc
                s1 += t
                s2 += t * t
                u = g[m] - ro
                e += u * u
            wij = w[i, i] if i == j else w[i, j] + w[j, i]
            e_1 += wij * e
            e_2 += wij * (M * s2 - s1 * s1)  # sum over member pairs k < m
    out[0] = e_1 / M - e_2 / (M * (M - 1))


@guvectorize("(d),(m,d),(d,d),(),(m),()->()", cache=True)
def _owvariogram_score_gufunc(obs, fct, w, ow, fw, p, out):
    M, D = fct.shape
    mode = 1 if p == 0.5 else 2 if p == 1.0 else 3 if p == 2.0 else 0
    c0 = abs(0.0) ** p
    cheap_diag = p != 0.0
    ft = np.empty((D, M), dtype=fct.dtype)
    for m in range(M):
        for d in range(D):
            ft[d, m] = fct[m, d]
    g = np.empty(M)
    S0 = 0.0
    c = 0
    for m in range(M):
        S0 += fw[m]
        if abs(fw[m]) > abs(fw[c]):
            c = m
    e_1 = 0.0
    e_2 = 0.0
    for i in range(D):
        for j in range(i, D):
            if i == j and cheap_diag:
                for m in range(M):
                    g[m] = c0 + float(ft[i, m] - ft[i, m])
            elif mode == 1:
                for m in range(M):
                    g[m] = math.sqrt(abs(float(ft[i, m] - ft[j, m])))
            elif mode == 2:
                for m in range(M):
                    g[m] = abs(float(ft[i, m] - ft[j, m]))
            elif mode == 3:
                for m in range(M):
                    t = float(ft[i, m] - ft[j, m])
                    g[m] = t * t
            else:
                for m in range(M):
                    g[m] = abs(float(ft[i, m] - ft[j, m])) ** p
            a = abs(float(obs[i] - obs[j]))
            if mode == 1:
                ro = math.sqrt(a)
            elif mode == 2:
                ro = a
            elif mode == 3:
                ro = a * a
            else:
                ro = a**p
            rc = g[c]
            s1 = 0.0
            s2 = 0.0
            e = 0.0
            for m in range(M):
                t = g[m] - rc
                at = fw[m] * t
                s1 += at
                s2 += at * t
                u = g[m] - ro
                e += fw[m] * (u * u)
            wij = w[i, i] if i == j else w[i, j] + w[j, i]
            e_1 += wij * e
            e_2 += wij * (S0 * s2 - s1 * s1)
    wbar = np.mean(fw)
    out[0] = e_1 * ow / (M * wbar) - e_2 * ow / (M**2 * wbar**2)


@guvectorize("(d),(m,d),(d,d),(),(m),()->()", cache=True)
def _vrvariogram_score_gufunc(obs, fct, w, ow, fw, p, out):
    M, D = fct.shape
    mode = 1 if p == 0.5 else 2 if p == 1.0 else 3 if p == 2.0 else 0
    c0 = abs(0.0) ** p
    cheap_diag = p != 0.0
    ft = np.empty((D, M), dtype=fct.dtype)
    for m in range(M):
        for d in range(D):
            ft[d, m] = fct[m, d]
    g = np.empty(M)
    S0 = 0.0
    c = 0
    for m in range(M):
        S0 += fw[m]
        if abs(fw[m]) > abs(fw[c]):
            c = m
    e_1 = 0.0
    e_2 = 0.0
    e_3_x = 0.0
    e_3_y = 0.0
    for i in range(D):
        for j in range(i, D):
            if i == j and cheap_diag:
                for m in range(M):
                    g[m] = c0 + float(ft[i, m] - ft[i, m])
            elif mode == 1:
                for m in range(M):
                    g[m] = math.sqrt(abs(float(ft[i, m] - ft[j, m])))
            elif mode == 2:
                for m in range(M):
                    g[m] = abs(float(ft[i, m] - ft[j, m]))
            elif mode == 3:
                for m in range(M):
                    t = float(ft[i, m] - ft[j, m])
                    g[m] = t * t
            else:
                for m in range(M):
                    g[m] = abs(float(ft[i, m] - ft[j, m])) ** p
            a = abs(float(obs[i] - obs[j]))
            if mode == 1:
                ro = math.sqrt(a)
            elif mode == 2:
                ro = a
            elif mode == 3:
                ro = a * a
            else:
                ro = a**p
            rc = g[c]
            s1 = 0.0
            s2 = 0.0
            e = 0.0
            e3 = 0.0
            for m in range(M):
                r = g[m]
                t = r - rc
                at = fw[m] * t
                s1 += at
                s2 += at * t
                u = r - ro
                e += fw[m] * (u * u)
                e3 += fw[m] * (r * r)
            wij = w[i, i] if i == j else w[i, j] + w[j, i]
            e_1 += wij * e
            e_2 += wij * (S0 * s2 - s1 * s1)
            e_3_x += wij * e3
            e_3_y += wij * (ro * ro)
    wbar = np.mean(fw)
    out[0] = e_1 * ow / M - e_2 / (M**2) + (e_3_x / M - e_3_y * ow) * (wbar - ow)


estimator_gufuncs = {
    "fair": lazy_gufunc_wrapper_mv(_variogram_score_fair_gufunc),
    "nrg": lazy_gufunc_wrapper_mv(_variogram_score_nrg_gufunc),
    "ownrg": lazy_gufunc_wrapper_mv(_owvariogram_score_gufunc),
    "vrnrg": lazy_gufunc_wrapper_mv(_vrvariogram_score_gufunc),
}

__all__ = [
    "_variogram_score_fair_gufunc",
    "_variogram_score_nrg_gufunc",
    "_owvariogram_score_gufunc",
    "_vrvariogram_score_gufunc",
]
