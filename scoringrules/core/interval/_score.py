import typing as tp

from scoringrules.backend import backends

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, Backend


def _interval_score(
    obs: "Array",
    lower: "Array",
    upper: "Array",
    alpha: "Array",
    backend: "Backend" = None,
) -> "Array":
    """Winkler or Interval Score for prediction interval PI[lower, upper] and observations."""
    # We don't need the backend here
    obs = obs[..., None]
    width = upper - lower
    above = obs > upper
    below = obs < lower
    W = width + (
        below * (2 / alpha) * (lower - obs) + above * (2 / alpha) * (obs - upper)
    )
    return W


def _weighted_interval_score(
    obs: "Array",
    median: "Array",
    lower: "Array",
    upper: "Array",
    alpha: "Array",
    weight_median: "Array",
    weight_alpha: "Array",
    backend: "Backend" = None,
) -> "Array":
    """Weighted Interval Score for prediction interval PI[lower, upper]."""
    B = backends.active if backend is None else backends[backend]
    K = weight_alpha.shape[0]
    IS = _interval_score(obs, lower, upper, alpha)
    WIS = (
        B.sum(IS * B.expand_dims(weight_alpha, axis=-2), axis=-1)
        + weight_median * median
    )
    WIS /= K + 1 / 2
    return WIS
