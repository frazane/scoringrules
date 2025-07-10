import numpy as np
from numba import guvectorize


@guvectorize(
    [
        "void(float32[:], float32[:,:], float32[:], float32,  float32[:])",
        "void(float64[:], float64[:,:], float64[:], float64, float64[:])",
    ],
    "(d),(m,d),(m),()->()",
)
def _variogram_score_gufunc(obs, fct, ens_w, p, out):
    M = fct.shape[-2]
    D = fct.shape[-1]
    out[0] = 0.0
    for i in range(D):
        for j in range(D):
            vfct = 0.0
            for m in range(M):
                vfct += ens_w[m] * abs(fct[m, i] - fct[m, j]) ** p
            vobs = abs(obs[i] - obs[j]) ** p
            out[0] += (vobs - vfct) ** 2


@guvectorize(
    [
        "void(float32[:], float32[:,:], float32, float32[:], float32[:], float32, float32[:])",
        "void(float64[:], float64[:,:], float64, float64[:], float64[:], float64, float64[:])",
    ],
    "(d),(m,d),(),(m),(m),()->()",
)
def _owvariogram_score_gufunc(obs, fct, ow, fw, ens_w, p, out):
    M = fct.shape[-2]
    D = fct.shape[-1]

    e_1 = 0.0
    e_2 = 0.0
    for k in range(M):
        for i in range(D):
            for j in range(D):
                rho1 = abs(fct[k, i] - fct[k, j]) ** p
                rho2 = abs(obs[i] - obs[j]) ** p
                e_1 += (rho1 - rho2) ** 2 * fw[k] * ow
        for m in range(k + 1, M):
            for i in range(D):
                for j in range(D):
                    rho1 = abs(fct[k, i] - fct[k, j]) ** p
                    rho2 = abs(fct[m, i] - fct[m, j]) ** p
                    e_2 += 2 * ((rho1 - rho2) ** 2) * fw[k] * fw[m] * ow

    wbar = np.mean(fw)

    out[0] = e_1 / (M * wbar) - 0.5 * e_2 / (M**2 * wbar**2)


@guvectorize(
    [
        "void(float32[:], float32[:,:], float32, float32[:], float32[:], float32, float32[:])",
        "void(float64[:], float64[:,:], float64, float64[:], float64[:], float64, float64[:])",
    ],
    "(d),(m,d),(),(m),(m),()->()",
)
def _vrvariogram_score_gufunc(obs, fct, ow, fw, ens_w, p, out):
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
                e_1 += (rho1 - rho2) ** 2 * fw[k] * ow
                e_3_x += (rho1) ** 2 * fw[k]
        for m in range(k + 1, M):
            for i in range(D):
                for j in range(D):
                    rho1 = abs(fct[k, i] - fct[k, j]) ** p
                    rho2 = abs(fct[m, i] - fct[m, j]) ** p
                    e_2 += 2 * ((rho1 - rho2) ** 2) * fw[k] * fw[m]

    e_3_x *= 1 / M
    wbar = np.mean(fw)
    e_3_y = 0.0
    for i in range(D):
        for j in range(D):
            rho1 = abs(obs[i] - obs[j]) ** p
            e_3_y += (rho1) ** 2 * ow

    out[0] = e_1 / M - 0.5 * e_2 / (M**2) + (e_3_x - e_3_y) * (wbar - ow)
