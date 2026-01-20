import numpy as np
from numba import guvectorize

from scoringrules.core.utils import lazy_gufunc_wrapper_mv


@lazy_gufunc_wrapper_mv
@guvectorize(
    [
        "void(float32[:], float32[:,:], float32[:])",
        "void(float64[:], float64[:,:], float64[:])",
    ],
    "(d),(m,d)->()",
    cache=True,
)
def _energy_score_nrg_gufunc(
    obs: np.ndarray,
    fct: np.ndarray,
    out: np.ndarray,
):
    """Compute the Energy Score for a finite ensemble."""
    M = fct.shape[0]

    e_1 = 0.0
    e_2 = 0.0
    for i in range(M):
        e_1 += float(np.linalg.norm(fct[i] - obs))
        for j in range(i + 1, M):
            e_2 += 2 * float(np.linalg.norm(fct[i] - fct[j]))

    out[0] = e_1 / M - 0.5 / (M**2) * e_2


@lazy_gufunc_wrapper_mv
@guvectorize(
    [
        "void(float32[:], float32[:,:], float32[:])",
        "void(float64[:], float64[:,:], float64[:])",
    ],
    "(d),(m,d)->()",
    cache=True,
)
def _energy_score_fair_gufunc(
    obs: np.ndarray,
    fct: np.ndarray,
    out: np.ndarray,
):
    """Compute the fair Energy Score for a finite ensemble."""
    M = fct.shape[0]

    e_1 = 0.0
    e_2 = 0.0
    for i in range(M):
        e_1 += float(np.linalg.norm(fct[i] - obs))
        for j in range(i + 1, M):
            e_2 += 2 * float(np.linalg.norm(fct[i] - fct[j]))

    out[0] = e_1 / M - 0.5 / (M * (M - 1)) * e_2


@lazy_gufunc_wrapper_mv
@guvectorize("(d),(m,d)->()")
def _energy_score_akr_gufunc(obs: np.ndarray, fct: np.ndarray, out: np.ndarray):
    """Compute the Energy Score for a finite ensemble using the approximate kernel representation."""
    M = fct.shape[0]

    e_1 = 0.0
    e_2 = 0.0
    for i in range(M):
        e_1 += float(np.linalg.norm(fct[i] - obs))
        e_2 += float(np.linalg.norm(fct[i] - fct[i - 1]))

    out[0] = e_1 / M - 0.5 * 1 / M * e_2


@lazy_gufunc_wrapper_mv
@guvectorize("(d),(m,d)->()")
def _energy_score_akr_circperm_gufunc(
    obs: np.ndarray, fct: np.ndarray, out: np.ndarray
):
    """Compute the Energy Score for a finite ensemble using the AKR with cyclic permutation."""
    M = fct.shape[0]

    e_1 = 0.0
    e_2 = 0.0
    for i in range(M):
        sigma_i = int((i + (M // 2)) % M)
        e_1 += float(np.linalg.norm(fct[i] - obs))
        e_2 += float(np.linalg.norm(fct[i] - fct[sigma_i]))

    out[0] = e_1 / M - 0.5 * 1 / M * e_2


@lazy_gufunc_wrapper_mv
@guvectorize("(d),(m,d),(),(m)->()")
def _owenergy_score_gufunc(
    obs: np.ndarray,
    fct: np.ndarray,
    ow: np.ndarray,
    fw: np.ndarray,
    out: np.ndarray,
):
    """Compute the Outcome-Weighted Energy Score for a finite ensemble."""
    M = fct.shape[0]

    e_1 = 0.0
    e_2 = 0.0
    for i in range(M):
        e_1 += float(np.linalg.norm(fct[i] - obs) * fw[i] * ow)
        for j in range(i + 1, M):
            e_2 += 2 * float(np.linalg.norm(fct[i] - fct[j]) * fw[i] * fw[j] * ow)

    wbar = np.mean(fw)

    out[0] = e_1 / (M * wbar) - 0.5 * e_2 / (M**2 * wbar**2)


@lazy_gufunc_wrapper_mv
@guvectorize("(d),(m,d),(),(m)->()")
def _vrenergy_score_gufunc(
    obs: np.ndarray,
    fct: np.ndarray,
    ow: np.ndarray,
    fw: np.ndarray,
    out: np.ndarray,
):
    """Compute the Vertically Re-scaled Energy Score for a finite ensemble."""
    M = fct.shape[0]

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


estimator_gufuncs = {
    "akr_circperm": _energy_score_akr_circperm_gufunc,
    "akr": _energy_score_akr_gufunc,
    "fair": _energy_score_fair_gufunc,
    "nrg": _energy_score_nrg_gufunc,
    "ownrg": _owenergy_score_gufunc,
    "vrnrg": _vrenergy_score_gufunc,
}

__all__ = [
    "_energy_score_akr_circperm_gufunc",
    "_energy_score_akr_gufunc",
    "_energy_score_fair_gufunc",
    "_energy_score_nrg_gufunc",
    "_owenergy_score_gufunc",
    "_vrenergy_score_gufunc",
]
