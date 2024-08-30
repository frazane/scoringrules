import numpy as np
from numba import guvectorize, vectorize


@vectorize(
    [
        "float64(float64, float64, float64, float64)",
        "float32(float32, float32, float32, float32)",
    ]
)
def _interval_score_gufunc(
    obs: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: np.ndarray,
):
    """Interval score or Winkler score."""
    return (
        (upper - lower)
        + (obs < lower) * (2 / alpha) * (lower - obs)
        + (obs > upper) * (2 / alpha) * (obs - upper)
    )


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:], float32[:], float32[:], float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:])",
    ],
    "(),(),(a),(a),(a),(),(a) ->()",
)
def _weighted_interval_score_gufunc(
    obs: np.ndarray,
    median: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: np.ndarray,
    weight_median: np.ndarray,
    weight_alpha: np.ndarray,
    out: np.ndarray,
):
    """Weighted Interval score (WIS)."""
    obs = obs[0]
    med = median[0]
    w_med = weight_median[0]
    K = alpha.shape[0]
    wis = 0
    for low, upp, a, w_a in zip(lower, upper, alpha, weight_alpha):  # noqa: B905
        ws_a = (
            (upp - low)
            + (obs < low) * (2 / a) * (low - obs)
            + (obs > upp) * (2 / a) * (obs - upp)
        )
        wis += w_a * ws_a
    wis += w_med * np.abs(obs - med)
    wis /= K + 1 / 2
    out[0] = wis
