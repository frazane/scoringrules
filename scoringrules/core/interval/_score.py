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
    B = backends.active if backend is None else backends[backend]
    width = upper - lower
    above = obs[:, None] > upper
    below = obs[:, None] < lower
    W = width + (
        below * (2 / alpha) * (lower - obs[:, None])
        + above * (2 / alpha) * (obs[:, None] - upper)
    )
    return B.mean(W, axis=-1)
