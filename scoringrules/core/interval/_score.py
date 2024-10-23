import typing as tp

from scoringrules.backend import backends

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, Backend


def interval_score(
    obs: "Array",
    lower: "Array",
    upper: "Array",
    alpha: "Array",
) -> "Array":
    """Winkler or Interval Score for prediction interval PI[lower, upper] and observations."""
    width = upper - lower
    above = obs > upper
    below = obs < lower
    W = width + below * (2 / alpha) * (lower - obs)
    W += above * (2 / alpha) * (obs - upper)
    return W


def weighted_interval_score(
    obs: "Array",
    median: "Array",
    lower: "Array",
    upper: "Array",
    alpha: "Array",
    w_median: "Array",
    w_alpha: "Array",
    backend: "Backend" = None,
) -> "Array":
    """Weighted Interval Score for prediction interval PI[lower, upper]."""
    B = backends.active if backend is None else backends[backend]
    K = w_alpha.shape[0]
    IS = interval_score(obs[..., None], lower, upper, alpha)
    WIS = B.sum(IS * w_alpha, axis=-1)
    WIS += w_median * median
    WIS /= K + 1 / 2
    return B.squeeze(WIS)
