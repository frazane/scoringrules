import typing as tp

from scoringrules.backend import backends

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike, Backend


def brier_score(
    obs: "ArrayLike", fct: "ArrayLike", backend: "Backend" = None
) -> "Array":
    """Compute the Brier Score for predicted probabilities of events."""
    B = backends.active if backend is None else backends[backend]
    obs, fct = map(B.asarray, (obs, fct))
    return (fct - obs) ** 2


def rps_score(
    obs: "ArrayLike",
    fct: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Compute the Ranked Probability Score for ordinal categorical forecasts."""
    B = backends.active if backend is None else backends[backend]
    return B.sum((B.cumsum(fct, axis=-1) - B.cumsum(obs, axis=-1)) ** 2, axis=-1)


def log_score(obs: "ArrayLike", fct: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Compute the Log Score for predicted probabilities of binary events."""
    B = backends.active if backend is None else backends[backend]
    return B.asarray(-B.log(B.abs(fct + obs - 1.0)))


def rls_score(
    obs: "ArrayLike",
    fct: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Compute the Ranked Logarithmic Score for ordinal categorical forecasts."""
    B = backends.active if backend is None else backends[backend]
    return B.sum(
        -B.log(B.abs(B.cumsum(fct, axis=-1) + B.cumsum(obs, axis=-1) - 1.0)),
        axis=-1,
    )
