import numpy as np
from numba import guvectorize

from scoringrules.core.utils import lazy_gufunc_wrapper_mv


@lazy_gufunc_wrapper_mv
@guvectorize("(d),(m,d),(d,d),(m),()->()")
def _variogram_score_gufunc_w(obs, fct, w, ens_w, p, out):
    M = fct.shape[-2]
    D = fct.shape[-1]
    out[0] = 0.0
    for i in range(D):
        for j in range(D):
            vfct = 0.0
            for m in range(M):
                vfct += ens_w[m] * abs(fct[m, i] - fct[m, j]) ** p
            vobs = abs(obs[i] - obs[j]) ** p
            out[0] += w[i, j] * (vobs - vfct) ** 2


@lazy_gufunc_wrapper_mv
@guvectorize("(d),(m,d),(d,d),(),(m),(m),()->()")
def _owvariogram_score_gufunc_w(obs, fct, w, ow, fw, ens_w, p, out):
    M = fct.shape[-2]
    D = fct.shape[-1]

    e_1 = 0.0
    e_2 = 0.0
    for k in range(M):
        for i in range(D):
            for j in range(D):
                rho1 = abs(fct[k, i] - fct[k, j]) ** p
                rho2 = abs(obs[i] - obs[j]) ** p
                e_1 += w[i, j] * (rho1 - rho2) ** 2 * fw[k] * ow * ens_w[k]
        for m in range(k + 1, M):
            for i in range(D):
                for j in range(D):
                    rho1 = abs(fct[k, i] - fct[k, j]) ** p
                    rho2 = abs(fct[m, i] - fct[m, j]) ** p
                    e_2 += w[i, j] * (
                        2
                        * ((rho1 - rho2) ** 2)
                        * fw[k]
                        * fw[m]
                        * ow
                        * ens_w[k]
                        * ens_w[m]
                    )

    wbar = np.sum(fw * ens_w)

    out[0] = e_1 / wbar - 0.5 * e_2 / (wbar**2)


@lazy_gufunc_wrapper_mv
@guvectorize("(d),(m,d),(d,d),(),(m),(m),()->()")
def _vrvariogram_score_gufunc_w(obs, fct, w, ow, fw, ens_w, p, out):
    M = fct.shape[-2]
    D = fct.shape[-1]

    e_1 = 0.0
    e_2 = 0.0
    e_3_x = 0.0
    for k in range(M):
        for i in range(D):
            for j in range(D):
                rho1 = abs(fct[k, i] - fct[k, j]) ** p
                rho2 = abs(obs[i] - obs[j]) ** p
                e_1 += w[i, j] * (rho1 - rho2) ** 2 * fw[k] * ow * ens_w[k]
                e_3_x += w[i, j] * (rho1) ** 2 * fw[k] * ens_w[k]
        for m in range(k + 1, M):
            for i in range(D):
                for j in range(D):
                    rho1 = abs(fct[k, i] - fct[k, j]) ** p
                    rho2 = abs(fct[m, i] - fct[m, j]) ** p
                    e_2 += w[i, j] * (
                        2 * ((rho1 - rho2) ** 2) * fw[k] * fw[m] * ens_w[k] * ens_w[m]
                    )

    wbar = np.sum(fw * ens_w)
    e_3_y = 0.0
    for i in range(D):
        for j in range(D):
            rho1 = abs(obs[i] - obs[j]) ** p
            e_3_y += w[i, j] * (rho1) ** 2 * ow

    out[0] = e_1 - 0.5 * e_2 + (e_3_x - e_3_y) * (wbar - ow)
