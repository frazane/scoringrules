"""Variogram score gufuncs with ensemble weights ``ens_w``.

Same structure as ``_gufuncs.py`` (see there): pairs ``i <= j``, sqrt/abs/square
for p equal to 0.5/1/2, a ``(D, M)`` copy of the members and the shifted
``S0 S2 - S1 ** 2`` identity for the spread term, loops written out per gufunc.
"""

import math

import numpy as np
from numba import guvectorize

from scoringrules.core.utils import lazy_gufunc_wrapper_mv


@guvectorize("(d),(m,d),(d,d),(m),()->()", cache=True)
def _variogram_score_nrg_gufunc_w(obs, fct, w, ens_w, p, out):
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
                s += ens_w[m] * g[m]
            vf[i, j] = vf[j, i] = s
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


@guvectorize("(d),(m,d),(d,d),(m),()->()", cache=True)
def _variogram_score_fair_gufunc_w(obs, fct, w, ens_w, p, out):
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
        S0 += ens_w[m]
        if abs(ens_w[m]) > abs(ens_w[c]):
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
                at = ens_w[m] * t
                s1 += at
                s2 += at * t
                u = g[m] - ro
                e += ens_w[m] * (u * u)
            wij = w[i, i] if i == j else w[i, j] + w[j, i]
            e_1 += wij * e
            e_2 += wij * (S0 * s2 - s1 * s1)  # sum over member pairs k < m
    fair_c = 1 - np.sum(ens_w**2)
    out[0] = e_1 - e_2 / fair_c


@guvectorize("(d),(m,d),(d,d),(),(m),(m),()->()", cache=True)
def _owvariogram_score_gufunc_w(obs, fct, w, ow, fw, ens_w, p, out):
    M, D = fct.shape
    mode = 1 if p == 0.5 else 2 if p == 1.0 else 3 if p == 2.0 else 0
    c0 = abs(0.0) ** p
    cheap_diag = p != 0.0
    ft = np.empty((D, M), dtype=fct.dtype)
    for m in range(M):
        for d in range(D):
            ft[d, m] = fct[m, d]
    g = np.empty(M)
    aw = np.empty(M)
    S0 = 0.0
    c = 0
    for m in range(M):
        aw[m] = fw[m] * ens_w[m]
        S0 += aw[m]
        if abs(aw[m]) > abs(aw[c]):
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
                at = aw[m] * t
                s1 += at
                s2 += at * t
                u = g[m] - ro
                e += aw[m] * (u * u)
            wij = w[i, i] if i == j else w[i, j] + w[j, i]
            e_1 += wij * e
            e_2 += wij * (S0 * s2 - s1 * s1)
    wbar = np.sum(fw * ens_w)
    out[0] = e_1 * ow / wbar - e_2 * ow / (wbar**2)


@guvectorize("(d),(m,d),(d,d),(),(m),(m),()->()", cache=True)
def _vrvariogram_score_gufunc_w(obs, fct, w, ow, fw, ens_w, p, out):
    M, D = fct.shape
    mode = 1 if p == 0.5 else 2 if p == 1.0 else 3 if p == 2.0 else 0
    c0 = abs(0.0) ** p
    cheap_diag = p != 0.0
    ft = np.empty((D, M), dtype=fct.dtype)
    for m in range(M):
        for d in range(D):
            ft[d, m] = fct[m, d]
    g = np.empty(M)
    aw = np.empty(M)
    S0 = 0.0
    c = 0
    for m in range(M):
        aw[m] = fw[m] * ens_w[m]
        S0 += aw[m]
        if abs(aw[m]) > abs(aw[c]):
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
                at = aw[m] * t
                s1 += at
                s2 += at * t
                u = r - ro
                e += aw[m] * (u * u)
                e3 += aw[m] * (r * r)
            wij = w[i, i] if i == j else w[i, j] + w[j, i]
            e_1 += wij * e
            e_2 += wij * (S0 * s2 - s1 * s1)
            e_3_x += wij * e3
            e_3_y += wij * (ro * ro)
    wbar = np.sum(fw * ens_w)
    out[0] = e_1 * ow - e_2 + (e_3_x - e_3_y * ow) * (wbar - ow)


estimator_gufuncs_w = {
    "fair": lazy_gufunc_wrapper_mv(_variogram_score_fair_gufunc_w),
    "nrg": lazy_gufunc_wrapper_mv(_variogram_score_nrg_gufunc_w),
    "ownrg": lazy_gufunc_wrapper_mv(_owvariogram_score_gufunc_w),
    "vrnrg": lazy_gufunc_wrapper_mv(_vrvariogram_score_gufunc_w),
}

__all__ = [
    "_variogram_score_fair_gufunc_w",
    "_variogram_score_nrg_gufunc_w",
    "_owvariogram_score_gufunc_w",
    "_vrvariogram_score_gufunc_w",
]
