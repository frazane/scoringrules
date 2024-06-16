import typing as tp

from scoringrules.core import brier

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike, Backend


def brier_score(
    observations: "ArrayLike",
    forecasts: "ArrayLike",
    /,
    *,
    backend: "Backend" = None,
) -> "Array":
    r"""
    Compute the Brier Score (BS).

    The BS is formulated as

    $$BS(f, y) = (f - y)^2,$$

    where $f \in [0, 1]$ is the predicted probability of an event and $y \in \{0, 1\}$ the actual outcome.

    Parameters
    ----------
    observations: NDArray
        Observed outcome, either 0 or 1.
    forecasts : NDArray
        Forecasted probabilities between 0 and 1.
    backend: str
        The name of the backend used for computations. Defaults to 'numpy'.

    Returns
    -------
    brier_score : NDArray
        The computed Brier Score.

    """
    return brier.brier_score(obs=observations, forecasts=forecasts, backend=backend)


__all__ = ["brier_score"]
