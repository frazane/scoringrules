import typing as tp

from scoringrules.backend import backends

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike, Backend

EPSILON = 1e-5


def brier_score(
    obs: "ArrayLike", fct: "ArrayLike", backend: "Backend" = None
) -> "Array":
    """Compute the Brier Score for predicted probabilities of events."""
    B = backends.active if backend is None else backends[backend]
    obs, fct = map(B.asarray, (obs, fct))

    if B.any(fct < 0.0) or B.any(fct > 1.0 + EPSILON):
        raise ValueError("Forecasted probabilities must be within 0 and 1.")

    if not set(B.unique_values(obs)) <= {0, 1}:
        raise ValueError("Observations must be 0, 1, or NaN.")

    return B.asarray((fct - obs) ** 2)
