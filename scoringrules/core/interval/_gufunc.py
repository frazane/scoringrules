import numpy as np
from numba import guvectorize


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:], float64[:], float64[:])",
    ],
    "(),(a),(a),(a)->(a)",
)
def _interval_score_gufunc(
    obs: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: np.ndarray,
    out: np.ndarray,
):
    """Interval score or Winkler score."""
    obs = obs[0]
    for i, (low, upp, a) in enumerate(zip(lower, upper, alpha)):  # noqa: B905
        out[i] = (
            (upp - low)
            + (obs < low) * (2 / a) * (low - obs)
            + (obs > upp) * (2 / a) * (obs - upp)
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
