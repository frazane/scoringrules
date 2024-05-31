import numpy as np
from numba import guvectorize


@guvectorize(
    [
        "void(float32[:], float32[:,:], float32[:])",
        "void(float64[:], float64[:,:], float64[:])",
    ],
    "(d),(m,d)->()",
)
def _energy_score_gufunc(
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
        for j in range(M):
            e_2 += float(np.linalg.norm(fct[i] - fct[j]))

    out[0] = e_1 / M - 0.5 / (M**2) * e_2


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
        for j in range(M):
            e_2 += float(np.linalg.norm(fct[i] - fct[j]) * fw[i] * fw[j] * ow)

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
        for j in range(M):
            e_2 += float(np.linalg.norm(fct[i] - fct[j]) * fw[i] * fw[j])

    wabs_x = wabs_x / M
    wbar = np.mean(fw)
    wabs_y = np.linalg.norm(obs) * ow

    out[0] = e_1 / M - 0.5 * e_2 / (M**2) + (wabs_x - wabs_y) * (wbar - ow)


@guvectorize(
    [
        "void(float32[:], float32[:,:], int32, float32[:])",
        "void(float64[:], float64[:,:], int64, float64[:])",
    ],
    "(d), (m, d), ()->()",
)
def _energy_score_iid_gufunc(
    obs: np.ndarray,
    fct: np.ndarray,
    seed: int,
    out: np.ndarray,
):
    """IID estimator for the energy score."""
    ## Set the seed
    np.random.seed(seed)

    M = fct.shape[0]
    K = int(M / 2)
    e_1 = 0
    e_2 = 0
    perm_fct = np.random.permutation(fct)
    for i in range(M):
        e_1 += float(np.linalg.norm(perm_fct[i] - obs))
    for i in range(K):
        e_2 += float(np.linalg.norm(perm_fct[i] - perm_fct[-i]))
    out[0] = e_1 / M - 0.5 / K * e_2


@guvectorize(
    [
        "void(float32[:], float32[:,:], int32, int32, float32[:])",
        "void(float64[:], float64[:,:], int64, int64, float64[:])",
    ],
    "(d), (m, d), (), ()->()",
)
def _energy_score_kband_gufunc(
    obs: np.ndarray,
    fct: np.ndarray,
    K: int,
    seed: int,
    out: np.ndarray,
):
    """K-Band estimator for the energy score."""
    ## Set the seed
    np.random.seed(seed)

    M = fct.shape[0]
    e_1 = 0
    e_2 = 0
    idx = np.random.permutation(np.arange(M, dtype=np.int64))

    for i in idx:
        e_1 += float(np.linalg.norm(fct[i] - obs))
    for k in range(K):
        idx_rolled = np.roll(idx, k + 1)
        for i, j in zip(idx, idx_rolled):  # noqa
            e_2 += float(np.linalg.norm(fct[i] - fct[j]))
    out[0] = e_1 / M - 0.5 / (K * M) * e_2
