import numpy as np
from numba import guvectorize


@guvectorize(
    [
        "void(float32[:], float32[:,:], float32[:], float32[:])",
        "void(float64[:], float64[:,:], float64[:], float64[:])",
    ],
    "(d),(m,d),(m)->()",
)
def _energy_score_gufunc(
    obs: np.ndarray,
    fct: np.ndarray,
    ens_w: np.ndarray,
    out: np.ndarray,
):
    """Compute the Energy Score for a finite ensemble."""
    M = fct.shape[0]

    e_1 = 0.0
    e_2 = 0.0
    for i in range(M):
        e_1 += float(np.linalg.norm(fct[i] - obs)) * ens_w[i]
        for j in range(i + 1, M):
            e_2 += 2 * float(np.linalg.norm(fct[i] - fct[j])) * ens_w[i] * ens_w[j]

    out[0] = e_1 - 0.5 * e_2


@guvectorize(
    [
        "void(float32[:], float32[:,:], float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:,:], float64[:], float64[:], float64[:])",
    ],
    "(d),(m,d),(),(m)->()",
)
def _owenergy_score_gufunc(
    obs: np.ndarray,
    fct: np.ndarray,
    ow: np.ndarray,
    fw: np.ndarray,
    out: np.ndarray,
):
    """Compute the Outcome-Weighted Energy Score for a finite ensemble."""
    M = fct.shape[0]
    ow = ow[0]

    e_1 = 0.0
    e_2 = 0.0
    for i in range(M):
        e_1 += float(np.linalg.norm(fct[i] - obs) * fw[i] * ow)
        for j in range(i + 1, M):
            e_2 += 2 * float(np.linalg.norm(fct[i] - fct[j]) * fw[i] * fw[j] * ow)

    wbar = np.mean(fw)

    out[0] = e_1 / (M * wbar) - 0.5 * e_2 / (M**2 * wbar**2)


@guvectorize(
    [
        "void(float32[:], float32[:,:], float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:,:], float64[:], float64[:], float64[:])",
    ],
    "(d),(m,d),(),(m)->()",
)
def _vrenergy_score_gufunc(
    obs: np.ndarray,
    fct: np.ndarray,
    ow: np.ndarray,
    fw: np.ndarray,
    out: np.ndarray,
):
    """Compute the Vertically Re-scaled Energy Score for a finite ensemble."""
    M = fct.shape[0]
    ow = ow[0]

    e_1 = 0.0
    e_2 = 0.0
    wabs_x = 0.0
    for i in range(M):
        e_1 += float(np.linalg.norm(fct[i] - obs) * fw[i] * ow)
        wabs_x += np.linalg.norm(fct[i]) * fw[i]
        for j in range(i + 1, M):
            e_2 += 2 * float(np.linalg.norm(fct[i] - fct[j]) * fw[i] * fw[j])

    wabs_x = wabs_x / M
    wbar = np.mean(fw)
    wabs_y = np.linalg.norm(obs) * ow

    out[0] = e_1 / M - 0.5 * e_2 / (M**2) + (wabs_x - wabs_y) * (wbar - ow)
