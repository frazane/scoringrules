import typing as tp

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array


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
    W = W + above * (2 / alpha) * (obs - upper)
    return W


def weighted_interval_score(
    obs: "Array",
    median: "Array",
    lower: "Array",
    upper: "Array",
    alpha: "Array",
    w_median: "Array",
    w_alpha: "Array",
    *,
    xp,
) -> "Array":
    """Weighted Interval Score for prediction interval PI[lower, upper]."""
    K = w_alpha.shape[0]
    IS = interval_score(obs[..., None], lower, upper, alpha)
    WIS = xp.sum(IS * w_alpha, axis=-1)
    WIS = WIS + w_median * median
    return WIS / (K + 1 / 2)
