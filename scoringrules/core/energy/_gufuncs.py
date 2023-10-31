import numpy as np
from numba import guvectorize
from numpy.typing import NDArray

_NDArray = NDArray[np.float32 | np.float64]


@guvectorize(
    [
        "void(float32[:,:], float32[:], float32[:])",
        "void(float64[:,:], float64[:], float64[:])",
    ],
    "(m,d),(d)->()",
)
def _energy_score_gufunc(
    forecasts: _NDArray,
    observations: _NDArray,
    out: _NDArray,
):
    """Compute the Energy Score for a finite ensemble."""
    M = forecasts.shape[0]

    e_1 = 0.0
    e_2 = 0.0
    for i in range(M):
        e_1 += float(np.linalg.norm(forecasts[i] - observations))
        for j in range(M):
            e_2 += float(np.linalg.norm(forecasts[i] - forecasts[j]))

    out[0] = e_1 / M - 0.5 / (M**2) * e_2


@guvectorize(
    [
        "void(float32[:,:], float32[:], float32[:], float32[:], float32[:])",
        "void(float64[:,:], float64[:], float64[:], float64[:], float64[:])",
    ],
    "(m,d),(d),(m),()->()",
)
def _owenergy_score_gufunc(
    forecasts: _NDArray,
    observations: _NDArray,
    fw: _NDArray,
    ow: _NDArray,
    out: _NDArray,
):
    """Compute the Outcome-Weighted Energy Score for a finite ensemble."""
    M = forecasts.shape[0]
    ow = ow[0]

    e_1 = 0.0
    e_2 = 0.0
    for i in range(M):
        e_1 += float(np.linalg.norm(forecasts[i] - observations) * fw[i] * ow)
        for j in range(M):
            e_2 += float(
                np.linalg.norm(forecasts[i] - forecasts[j]) * fw[i] * fw[j] * ow
            )

    wbar = np.mean(fw)

    out[0] = e_1 / (M * wbar) - 0.5 * e_2 / (M**2 * wbar**2)


@guvectorize(
    [
        "void(float32[:,:], float32[:], float32[:], float32[:], float32[:])",
        "void(float64[:,:], float64[:], float64[:], float64[:], float64[:])",
    ],
    "(m,d),(d),(m),()->()",
)
def _vrenergy_score_gufunc(
    forecasts: _NDArray,
    observations: _NDArray,
    fw: _NDArray,
    ow: _NDArray,
    out: _NDArray,
):
    """Compute the Vertically Re-scaled Energy Score for a finite ensemble."""
    M = forecasts.shape[0]
    ow = ow[0]

    e_1 = 0.0
    e_2 = 0.0
    wabs_x = 0.0
    for i in range(M):
        e_1 += float(np.linalg.norm(forecasts[i] - observations) * fw[i] * ow)
        wabs_x += np.linalg.norm(forecasts[i]) * fw[i]
        for j in range(M):
            e_2 += float(np.linalg.norm(forecasts[i] - forecasts[j]) * fw[i] * fw[j])

    wabs_x = wabs_x / M
    wbar = np.mean(fw)
    wabs_y = np.linalg.norm(observations) * ow

    out[0] = e_1 / M - 0.5 * e_2 / (M**2) + (wabs_x - wabs_y) * (wbar - ow)
