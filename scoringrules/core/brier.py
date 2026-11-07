import typing as tp

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike


def brier_score(obs: "ArrayLike", fct: "ArrayLike") -> "Array":
    """Compute the Brier Score for predicted probabilities of events."""
    return (fct - obs) ** 2


def rps_score(obs: "ArrayLike", fct: "ArrayLike", *, xp) -> "Array":
    """Compute the Ranked Probability Score for ordinal categorical forecasts."""
    return xp.sum(
        (xp.cumulative_sum(fct, axis=-1) - xp.cumulative_sum(obs, axis=-1)) ** 2,
        axis=-1,
    )


def log_score(obs: "ArrayLike", fct: "ArrayLike", *, xp) -> "Array":
    """Compute the Log Score for predicted probabilities of binary events."""
    # asarray: for 0-d numpy input the namespace's scalar-tolerant log/abs
    # return a Python float
    return xp.asarray(-xp.log(xp.abs(fct + obs - 1.0)))


def rls_score(obs: "ArrayLike", fct: "ArrayLike", *, xp) -> "Array":
    """Compute the Ranked Logarithmic Score for ordinal categorical forecasts."""
    return xp.sum(
        -xp.log(
            xp.abs(
                xp.cumulative_sum(fct, axis=-1) + xp.cumulative_sum(obs, axis=-1) - 1.0
            )
        ),
        axis=-1,
    )
