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
    Brier Score

    The Brier Score is defined as

    .. math::
       \text{BS}(F, y) = (F - y)^2,

    where :math:`F \in [0, 1]` is the predicted probability of an event and :math:`y \in \{0, 1\}`
    is the outcome [1]_.

    Parameters
    ----------
    obs : array_like
        Observed outcomes, either 0 or 1.
    fct : array_like
        Forecast probabilities, between 0 and 1.
    backend : str
        The name of the backend used for computations. Default is 'numpy'.

    Returns
    -------
    score : array_like
        The computed Brier Score.

    References
    ----------
    .. [1] Brier, G. W. (1950).
        Verification of forecasts expressed in terms of probability.
        Monthly Weather Review, 78, 1-3.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.brier_score(1, 0.2)
    0.64000
    """
    return brier.brier_score(obs=obs, fct=fct, backend=backend)


def rps_score(
    obs: "ArrayLike",
    fct: "ArrayLike",
    /,
    k_axis: int = -1,
    *,
    onehot: bool = False,
    backend: "Backend" = None,
) -> "Array":
    r"""
    (Discrete) Ranked Probability Score (RPS)

    Suppose the outcome corresponds to one of :math:`K` ordered categories. The RPS is defined as

    .. math::
       \text{RPS}(F, y) = \sum_{k=1}^{K}(\tilde{F}_{k} - \tilde{y}_{k})^2,

    where :math:`F \in [0, 1]^{K}` is a vector of length :math:`K`, containing forecast probabilities
    that each of the :math:`K` categories will occur, with :math:`\sum_{k=1}^{K} F_{k} = 1` and
    :math:`\tilde{F}_{k} = \sum_{i=1}^{k} F_{i}` for all :math:`k = 1, \dots, K`, and where
    :math:`y \in \{1, \dots, K\}` is the category that occurs, with :math:`\tilde{y}_{k} = 1\{y \le i\}`
    for all :math:`k = 1, \dots, K` [1]_.

    The outcome can alternatively be interpreted as a vector :math:`y \in \{0, 1\}^K` of length :math:`K`, with the
    :math:`k`-th element equal to one if the :math:`k`-th category occurs, and zero otherwise.
    Using this one-hot encoding, the RPS is defined analogously to as above, but with
    :math:`\tilde{y}_{k} = \sum_{i=1}^{k} y_{i}`.

    Parameters
    ----------
    obs : array_like
        Category that occurs. Or array of 0's and 1's corresponding to unobserved and
        observed categories if `onehot=True`.
    fct : array
        Array of forecast probabilities for each category.
    k_axis: int
        The axis of `obs` and `fct` corresponding to the categories. Default is the last axis.
    onehot: bool
        Boolean indicating whether the observation is the category that occurs or a onehot
        encoded vector of 0's and 1's. Default is False.
    backend : str
        The name of the backend used for computations. Default is 'numpy'.

    Returns
    -------
    score: array_like
        The computed Ranked Probability Score.

    References
    ----------
    .. [1] Epstein, E. S. (1969).
        A scoring system for probability forecasts of ranked categories.
        Journal of Applied Meteorology, 8, 985-987.
        https://www.jstor.org/stable/26174707.

    Examples
    --------
    >>> import scoringrules as sr
    >>> import numpy as np
    >>> fct = np.array([0.1, 0.2, 0.3, 0.4])
    >>> obs = 3
    >>> sr.rps_score(obs, fct)
    0.25999999999999995
    >>> obs = np.array([0, 0, 1, 0])
    >>> sr.rps_score(obs, fct, onehot=True)
    0.25999999999999995
    """
    B = backends.active if backend is None else backends[backend]
    fct = B.asarray(fct)

    if k_axis != -1:
        fct = B.moveaxis(fct, k_axis, -1)
        if onehot:
            obs = B.moveaxis(obs, k_axis, -1)

    return brier.rps_score(obs=obs, fct=fct, onehot=onehot, backend=backend)


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
