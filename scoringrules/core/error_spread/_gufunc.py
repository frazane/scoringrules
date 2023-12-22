import math

import numpy as np
from numba import guvectorize

INV_SQRT_PI = 1 / np.sqrt(np.pi)
EPSILON = 1e-6


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:])",
    ],
    "(n),()->()",
)
def _error_spread_score_gufunc(
    forecasts: np.ndarray, observation: np.ndarray, out: np.ndarray
):
    """Error spread score."""
    M = forecasts.shape[0]
    m = 0.0
    s = 0.0
    g = 0.0
    e = 0.0

    for i in range(M):
        m += forecasts[i]
    m /= M

    for i in range(M):
        s += (forecasts[i] - m) ** 2
    s /= (M - 1)
    s = np.sqrt(s)

    for i in range(M):
        g += ((forecasts[i] - m) / s) ** 3
    g /= ((M - 1) * (M - 2))

    for i in range(M):
        e += forecasts[i] - observation[i]

    out[0] = (s**2 - e**2 - e * s * g) ** 2

