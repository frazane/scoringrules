from typing import TypeVar

from scoringrules.backend import backends
from scoringrules.core.typing import ArrayLike, Array
from scoringrules.core import brier


def brier_score(
    forecasts: ArrayLike, observations: ArrayLike, /, *, backend="numpy"
) -> Array:
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
    return brier.brier_score(forecasts, observations, backend=backend)


__all__ = ["brier_score"]
