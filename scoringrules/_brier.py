import typing as tp

from scoringrules.backend import backends
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

    $$ BS(f, y) = (f - y)^2, $$

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
    return brier.brier_score(obs=observations, fct=forecasts, backend=backend)


def rps_score(
    observations: "ArrayLike",
    forecasts: "ArrayLike",
    /,
    axis: int = -1,
    *,
    backend: "Backend" = None,
) -> "Array":
    r"""
    Compute the (Discrete) Ranked Probability Score (RPS).

    Suppose the outcome corresponds to one of $K$ ordered categories. The RPS is defined as

    $$ RPS(f, y) = \sum_{k=1}^{K}(\tilde{f}_{k} - \tilde{y}_{k})^2, $$

    where $f \in [0, 1]^{K}$ is a vector of length $K$ containing forecast probabilities
    that each of the $K$ categories will occur, and $y \in \{0, 1\}^{K}$ is a vector of
    length $K$, with the $k$-th element equal to one if the $k$-th category occurs. We
    have $\sum_{k=1}^{K} y_{k} = \sum_{k=1}^{K} f_{k} = 1$, and, for $k = 1, \dots, K$,
    $\tilde{y}_{k} = \sum_{i=1}^{k} y_{i}$ and $\tilde{f}_{k} = \sum_{i=1}^{k} f_{i}$.

    Parameters
    ----------
    observations:
        Array of 0's and 1's corresponding to unobserved and observed categories
    forecasts :
        Array of forecast probabilities for each category.
    axis: int
        The axis corresponding to the categories. Default is the last axis.
    backend: str
        The name of the backend used for computations. Defaults to 'numpy'.

    Returns
    -------
    score:
        The computed Ranked Probability Score.

    """
    B = backends.active if backend is None else backends[backend]
    forecasts = B.asarray(forecasts)

    if axis != -1:
        forecasts = B.moveaxis(forecasts, axis, -1)

    return brier.rps_score(obs=observations, fct=forecasts, backend=backend)


__all__ = [
    "brier_score",
    "rps_score",
]
