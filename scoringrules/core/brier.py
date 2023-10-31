import typing as tp

from scoringrules.backend import backends

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike

EPSILON = 1e-5


def brier_score(
    fcts: "ArrayLike", obs: "ArrayLike", backend: str | None = None
) -> "Array":
    """Compute the Brier Score for predicted probabilities of events."""
    B = backends.active if backend is None else backends[backend]
    fcts, obs = map(B.asarray, (fcts, obs))

    if B.any(fcts < 0.0) or B.any(fcts > 1.0 + EPSILON):
        raise ValueError("Forecasted probabilities must be within 0 and 1.")

    if not set(B.unique_values(obs)) <= {0, 1}:
        raise ValueError("Observations must be 0, 1, or NaN.")

    return B.asarray((fcts - obs) ** 2)
