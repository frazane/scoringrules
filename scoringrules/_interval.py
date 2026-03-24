import typing as tp
from typing import Optional

from scoringrules.backend import backends
from scoringrules.core import interval

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike, Backend


def interval_score(
    obs: "ArrayLike",
    lower: "ArrayLike",
    upper: "ArrayLike",
    alpha: "ArrayLike",
    *,
    backend: "Backend" = None,
) -> "Array":
    r"""Compute the Interval Score or Winkler Score.

    The interval score
    [(Gneiting & Raftery, 2012)](https://doi.org/10.1198/016214506000001437)
    is defined as

    .. math::
        \text{IS} =
            \begin{cases}
            (u - l) + \frac{2}{\alpha}(l - y)  & \text{for } y < l \\
            (u - l)                            & \text{for } l \leq y \leq u \\
            (u - l) + \frac{2}{\alpha}(y - u)  & \text{for } y > u. \\
            \end{cases}

    for an :math:`1 - \alpha` prediction interval of :math:`[l, u]` and the true value :math:`y`.

    Parameters
    ----------
    obs : array_like
        The observations as a scalar or array of values.
    lower : array_like
        The predicted lower bound of the PI as a scalar or array of values.
    upper : array_like
        The predicted upper bound of the PI as a scalar or array of values.
    alpha : array_like
        The 1 - alpha level for the PI as a scalar or array of values.
    backend : str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    interval_score : array_like
        Array with the interval score for the input values.

    Raises
    ------
    ValueError:
        If the lower and upper bounds do not have the same
        shape or if the number of PIs does not match the number of alpha levels.

    Notes
    -----
    Given an `obs` array of shape `(...,)`, in the case when multiple PIs are
    evaluated `alpha` is an array of shape `(K,)`, then `lower` and `upper` must
    have shape `(...,K)` and the output will have shape `(...,K)`.

    Examples
    --------
    >>> import numpy as np
    >>> import scoringrules as sr
    >>> sr.interval_score(0.1, 0.0, 0.4, 0.5)
    0.4

    >>> sr.interval_score(
    ...     obs=np.array([0.1, 0.2, 0.3]),
    ...     lower=np.array([0.0, 0.1, 0.2]),
    ...     upper=np.array([0.4, 0.3, 0.5]),
    ...     alpha=0.5,
    ... )
    array([0.4, 0.2, 0.4])

    >>> sr.interval_score(
    ...     obs=np.random.uniform(size=(10,)),
    ...     lower=np.ones((10,5)) * 0.2,
    ...     upper=np.ones((10,5)) * 0.8,
    ...     alpha=np.linspace(0.1, 0.9, 5),
    ... ).shape
    (10, 5)
    """
    B = backends.active if backend is None else backends[backend]
    obs, lower, upper, alpha = map(B.asarray, (obs, lower, upper, alpha))

    if lower.shape != upper.shape:
        raise ValueError(
            "The lower and upper bounds must have the same shape."
            f" Got lower {lower.shape} and upper {upper.shape}."
        )

    if alpha.ndim == 1:
        obs = obs[..., None]
        if (lower.shape[-1] != alpha.shape[0]) or (upper.shape[-1] != alpha.shape[0]):
            raise ValueError(
                "The number of PIs does not match the number of alpha levels."
                f" Got lower and upper of shapes {lower.shape}"
                f" for alpha of shape {alpha.shape}."
            )

    res = interval.interval_score(obs, lower, upper, alpha)
    return B.squeeze(res)


def weighted_interval_score(
    obs: "ArrayLike",
    median: "Array",
    lower: "Array",
    upper: "Array",
    alpha: "Array",
    /,
    w_median: Optional[float] = None,
    w_alpha: Optional["Array"] = None,
    *,
    backend: "Backend" = None,
) -> "Array":
    r"""Compute the weighted interval score (WIS).

    The WIS [(Bracher et al., 2022)](https://doi.org/10.1371/journal.pcbi.1008618)
    is defined as

    .. math::
        \text{WIS}_{\alpha_{0:K}}(F, y) = \frac{1}{K+0.5}(w_0 \times |y - m|
        + \sum_{k=1}^K (w_k \times IS_{\alpha_k}(F, y)))

    where :math:`m` denotes the median prediction, :math:`w_0` denotes the weight of the
    median prediction, :math:`IS_{\alpha_k}(F, y)` denotes the interval score for the
    :math:`1 - \alpha` prediction interval and :math:`w_k` is the according weight.
    The WIS is calculated for a set of (central) PIs and the predictive median.
    The weights are an optional parameter and default weight is the canonical
    weight :math:`w_k = \frac{2}{\alpha_k}` and :math:`w_0 = 0.5`.
    For these weights, it holds that:

    .. math::
        \text{WIS_{\alpha_{0:K}}(F, y) \approx \text{CRPS}(F, y).

    Parameters
    ----------
    obs : array_like
        The observations as a scalar or array of shape `(...,)`.
    median : array_like
        The predicted median of the distribution as a scalar or array of shape `(...,)`.
    lower : array_like
        The predicted lower bound of the PI. If `alpha` is an array of shape `(K,)`,
        `lower` must have shape `(...,K)`.
    upper : array_like
        The predicted upper bound of the PI. If `alpha` is an array of shape `(K,)`,
        `upper` must have shape `(...,K)`.
    alpha : array_like
        The 1 - alpha level for the prediction intervals as an array of shape `(K,)`.
    w_median : float
        The weight for the median prediction. Defaults to 0.5.
    w_alpha : array_like
        The weights for the PI. Defaults to `2/alpha`.
    backend : str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    score:
        An array of interval scores with the same shape as `obs`.
    """
    if w_median is None:
        w_median = 0.5
    if w_alpha is None:
        w_alpha = alpha / 2

    B = backends.active if backend is None else backends[backend]
    args = map(
        B.asarray,
        (obs, median, lower, upper, alpha, w_median, w_alpha),
    )

    if B.name == "numba":
        return interval._weighted_interval_score_gufunc(*args)
    else:
        return interval.weighted_interval_score(*args, backend=backend)
