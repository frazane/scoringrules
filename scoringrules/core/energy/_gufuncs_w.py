import numpy as np
from numba import guvectorize

from scoringrules.core.utils import lazy_gufunc_wrapper_mv


@guvectorize("(d),(m,d),(m)->()")
def _energy_score_nrg_gufunc_w(
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


@guvectorize("(d),(m,d),(m)->()")
def _energy_score_fair_gufunc_w(
    obs: np.ndarray,
    fct: np.ndarray,
    ens_w: np.ndarray,
    out: np.ndarray,
):
    """Compute the fair Energy Score for a finite ensemble."""
    M = fct.shape[0]

    e_1 = 0.0
    e_2 = 0.0
    for i in range(M):
        e_1 += float(np.linalg.norm(fct[i] - obs)) * ens_w[i]
        for j in range(i + 1, M):
            e_2 += 2 * float(np.linalg.norm(fct[i] - fct[j])) * ens_w[i] * ens_w[j]

    fair_c = 1 - np.sum(ens_w * ens_w)

    out[0] = e_1 - 0.5 * e_2 / fair_c


@guvectorize("(d),(m,d),(m)->()")
def _energy_score_akr_gufunc_w(
    obs: np.ndarray, fct: np.ndarray, ens_w: np.ndarray, out: np.ndarray
):
    """Compute the Energy Score for a finite ensemble using the approximate kernel representation."""
    M = fct.shape[0]

    e_1 = 0.0
    e_2 = 0.0
    for i in range(M):
        e_1 += float(np.linalg.norm(fct[i] - obs)) * ens_w[i]
        e_2 += float(np.linalg.norm(fct[i] - fct[i - 1])) * ens_w[i]

    out[0] = e_1 - 0.5 * e_2


@guvectorize("(d),(m,d),(m)->()")
def _energy_score_akr_circperm_gufunc_w(
    obs: np.ndarray, fct: np.ndarray, ens_w: np.ndarray, out: np.ndarray
):
    """Compute the Energy Score for a finite ensemble using the AKR with cyclic permutation."""
    M = fct.shape[0]

    e_1 = 0.0
    e_2 = 0.0
    for i in range(M):
        sigma_i = int((i + (M // 2)) % M)
        e_1 += float(np.linalg.norm(fct[i] - obs)) * ens_w[i]
        e_2 += float(np.linalg.norm(fct[i] - fct[sigma_i])) * ens_w[i]

    out[0] = e_1 - 0.5 * e_2


@guvectorize("(d),(m,d),(),(m),(m)->()")
def _owenergy_score_gufunc_w(
    obs: np.ndarray,
    fct: np.ndarray,
    ow: np.ndarray,
    fw: np.ndarray,
    ens_w: np.ndarray,
    out: np.ndarray,
):
    """Compute the Outcome-Weighted Energy Score for a finite ensemble."""
    M = fct.shape[0]
    ow = ow[0]

    e_1 = 0.0
    e_2 = 0.0
    for i in range(M):
        e_1 += float(np.linalg.norm(fct[i] - obs) * fw[i] * ow) * ens_w[i]
        for j in range(i + 1, M):
            e_2 += (
                2
                * float(np.linalg.norm(fct[i] - fct[j]) * fw[i] * fw[j] * ow)
                * ens_w[i]
                * ens_w[j]
            )

    wbar = np.mean(fw)

    out[0] = e_1 / (wbar) - 0.5 * e_2 / (wbar**2)


@guvectorize("(d),(m,d),(),(m),(m)->()")
def _vrenergy_score_gufunc_w(
    obs: np.ndarray,
    fct: np.ndarray,
    ow: np.ndarray,
    fw: np.ndarray,
    ens_w: np.ndarray,
    out: np.ndarray,
):
    """Compute the Vertically Re-scaled Energy Score for a finite ensemble."""
    M = fct.shape[0]
    ow = ow[0]

    e_1 = 0.0
    e_2 = 0.0
    wabs_x = 0.0
    for i in range(M):
        e_1 += float(np.linalg.norm(fct[i] - obs) * fw[i] * ow) * ens_w[i]
        wabs_x += np.linalg.norm(fct[i]) * fw[i] * ens_w[i]
        for j in range(i + 1, M):
            e_2 += (
                2
                * float(np.linalg.norm(fct[i] - fct[j]) * fw[i] * fw[j])
                * ens_w[i]
                * ens_w[j]
            )

    wbar = np.sum(fw * ens_w)
    wabs_y = np.linalg.norm(obs) * ow

    out[0] = e_1 - 0.5 * e_2 + (wabs_x - wabs_y) * (wbar - ow)


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
