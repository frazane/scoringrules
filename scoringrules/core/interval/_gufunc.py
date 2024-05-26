import numpy as np
from numba import guvectorize


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:], float64[:], float64[:])",
    ],
    "(),(a),(a),(a)->()",
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
    A = alpha.shape[0]
    ws = 0
    for low, upp, a in zip(lower, upper, alpha):  # noqa: B905
        width = upp - low
        ws += (
            width
            + (obs < low) * (2 / a) * (low - obs)
            + (obs > upp) * (2 / a) * (obs - upp)
        )
    out[0] = ws / A
