from typing import TypeVar

from scoringrules.backend import backends as srb
from scoringrules.backend.arrayapi import Array

ArrayLike = TypeVar("ArrayLike", Array, float)


def brier_score(
    forecasts: ArrayLike, observations: ArrayLike, /, *, backend="numpy"
) -> ArrayLike:
    r"""
    Compute the Brier Score (BS).

    The BS is formulated as

    $$BS(f, y) = (f - y)^2,$$

    where $f \in [0, 1]$ is the predicted probability of an event and $y \in \{0, 1\}$ the actual outcome.

    Parameters
    ----------
    forecasts : NDArray
        Forecasted probabilities between 0 and 1.
    observations: NDArray
        Observed outcome, either 0 or 1.
    backend: str
        The name of the backend used for computations. Defaults to 'numpy'.

    Returns
    -------
    brier_score : NDArray
        The computed Brier Score.

    """
    return srb[backend].brier_score(forecasts, observations)


__all__ = ["brier_score"]
