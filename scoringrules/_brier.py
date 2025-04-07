import typing as tp

from scoringrules.backend import backends
from scoringrules.core import brier

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike, Backend


def brier_score(
    obs: "ArrayLike",
    fct: "ArrayLike",
    /,
    *,
    backend: "Backend" = None,
) -> "Array":
    r"""
    Compute the Brier Score (BS).

    The BS is formulated as

    .. math::
       BS(f, y) = (f - y)^2,

    where :math:`f \in [0, 1]` is the predicted probability of an event and :math:`y \in \{0, 1\}` the actual outcome.

    Parameters
    ----------
    obs : array_like
        Observed outcome, either 0 or 1.
    fct : array_like
        Forecasted probabilities between 0 and 1.
    backend : str
        The name of the backend used for computations. Defaults to 'numpy'.

    Returns
    -------
    brier_score : array_like
        The computed Brier Score.

    """
    return brier.brier_score(obs=obs, fct=fct, backend=backend)


def rps_score(
    obs: "ArrayLike",
    fct: "ArrayLike",
    /,
    k_axis: int = -1,
    *,
    backend: "Backend" = None,
) -> "Array":
    r"""
    Compute the (Discrete) Ranked Probability Score (RPS).

    Suppose the outcome corresponds to one of :math:`K` ordered categories. The RPS is defined as

    .. math::
       RPS(f, y) = \sum_{k=1}^{K}(\tilde{f}_{k} - \tilde{y}_{k})^2,

    where :math:`f \in [0, 1]^{K}` is a vector of length :math:`K` containing forecast probabilities
    that each of the :math:`K` categories will occur, and :math:`y \in \{0, 1\}^{K}` is a vector of
    length :math:`K`, with the :math:`k`-th element equal to one if the :math:`k`-th category occurs. We
    have :math:`\sum_{k=1}^{K} y_{k} = \sum_{k=1}^{K} f_{k} = 1`, and, for :math:`k = 1, \dots, K`,
    :math:`\tilde{y}_{k} = \sum_{i=1}^{k} y_{i}` and :math:`\tilde{f}_{k} = \sum_{i=1}^{k} f_{i}`.

    Parameters
    ----------
    obs : array_like
        Array of 0's and 1's corresponding to unobserved and observed categories
    forecasts :
        Array of forecast probabilities for each category.
    k_axis: int
        The axis corresponding to the categories. Default is the last axis.
    backend : str
        The name of the backend used for computations. Defaults to 'numpy'.

    Returns
    -------
    score:
        The computed Ranked Probability Score.

    """
    B = backends.active if backend is None else backends[backend]
    fct = B.asarray(fct)

    if k_axis != -1:
        fct = B.moveaxis(fct, k_axis, -1)

    return brier.rps_score(obs=obs, fct=fct, backend=backend)


def log_score(
    obs: "ArrayLike",
    fct: "ArrayLike",
    /,
    *,
    backend: "Backend" = None,
) -> "Array":
    r"""
    Compute the Logarithmic Score (LS) for probability forecasts for binary outcomes.

    The LS is formulated as

    .. math::
       LS(f, y) = -\log|f + y - 1|,

    where :math:`f \in [0, 1]` is the predicted probability of an event and :math:`y \in \{0, 1\}` the actual outcome.

    Parameters
    ----------
    obs : array_like
        Observed outcome, either 0 or 1.
    fct : array_like
        Forecasted probabilities between 0 and 1.
    backend : str
        The name of the backend used for computations. Defaults to 'numpy'.

    Returns
    -------
    score:
        The computed Log Score.

    """
    return brier.log_score(obs=obs, fct=fct, backend=backend)


def rls_score(
    obs: "ArrayLike",
    fct: "ArrayLike",
    /,
    k_axis: int = -1,
    *,
    backend: "Backend" = None,
) -> "Array":
    r"""
    Compute the (Discrete) Ranked Logarithmic Score (RLS).

    Suppose the outcome corresponds to one of :math:`K` ordered categories. The RLS is defined as

    .. math::
       RPS(f, y) = -\sum_{k=1}^{K} \log|\tilde{f}_{k} + \tilde{y}_{k} - 1|,

    where :math:`f \in [0, 1]^{K}` is a vector of length :math:`K` containing forecast probabilities
    that each of the :math:`K` categories will occur, and :math:`y \in \{0, 1\}^{K}` is a vector of
    length :math:`K`, with the :math:`k`-th element equal to one if the :math:`k`-th category occurs. We
    have :math:`\sum_{k=1}^{K} y_{k} = \sum_{k=1}^{K} f_{k} = 1`, and, for :math:`k = 1, \dots, K`,
    :math:`\tilde{y}_{k} = \sum_{i=1}^{k} y_{i}` and :math:`\tilde{f}_{k} = \sum_{i=1}^{k} f_{i}`.

    Parameters
    ----------
    obs : array_like
        Observed outcome, either 0 or 1.
    fct : array_like
        Forecasted probabilities between 0 and 1.
    k_axis: int
        The axis corresponding to the categories. Default is the last axis.
    backend : str
        The name of the backend used for computations. Defaults to 'numpy'.

    Returns
    -------
    score : array_like
        The computed Ranked Logarithmic Score.

    """
    B = backends.active if backend is None else backends[backend]
    fct = B.asarray(fct)

    if k_axis != -1:
        fct = B.moveaxis(fct, k_axis, -1)

    return brier.rls_score(obs=obs, fct=fct, backend=backend)


__all__ = [
    "brier_score",
    "rps_score",
    "log_score",
    "rls_score",
]
