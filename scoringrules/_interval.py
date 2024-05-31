import typing as tp
from typing import Optional, Union

from scoringrules.backend import backends
from scoringrules.core import interval

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike, Backend


def interval_score(
    observations: "ArrayLike",
    lower: "Array",
    upper: "Array",
    alpha: Union[float, "Array"],
    /,
    axis: int = -1,
    *,
    backend: "Backend" = None,
) -> "Array":
    r"""Compute the Interval Score or Winkler Score [(Bracher J, Ray EL, Gneiting T, Reich NG, 2022)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008618) for 1 - $\alpha$ prediction intervals PI = [lower, upper].

    The interval score is defined as

    $\text{IS} = \begin{cases}
    (u - l) + 2\frac{2}{\alpha}(l - y)  & \text{for } y < l \\
    (u - l)                             & \text{for } l \leq y \leq u \\
    (u - l) + \frac{2}{\alpha}(y - u)   & \text{for } y > u. \\
    \end{cases}$

    for an $1 - \alpha$ PI of $[l, u]$ and the true value $y$.

    Note
    ----
    Note that alpha can be a float or an array of coverages.
    In the case alpha is a float, the output will have the same shape as the observations and we assume that shape of observations,
    upper and lower is the same. In case alpha is a vector, the function will broadcast observations accordingly.

    Parameters
    ----------
    observations: ArrayLike
        The observed values.
    lower: Array
        The predicted lower bound of the prediction interval.
    upper: Array
        The predicted upper bound of the prediction interval.
    alpha: Union[float, Array]
        The 1 - alpha level for the prediction interval.
    axis: int
        The axis corresponding to the ensemble. Default is the last axis.
    backend: str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    score: Array
        An array of interval scores for each prediction interval, which should be averaged to get meaningful values.
    """
    B = backends.active if backend is None else backends[backend]
    single_alpha = isinstance(alpha, float)

    observations, lower, upper = map(B.asarray, (observations, lower, upper))

    if axis != -1:
        lower = B.moveaxis(lower, axis, -1)
        upper = B.moveaxis(upper, axis, -1)

    if single_alpha:
        if B.name == "numba":
            return interval._interval_score_gufunc(observations, lower, upper, alpha)

        return interval._interval_score(
            observations, lower, upper, alpha, backend=backend
        )

    else:
        alpha = B.asarray(alpha)

        if B.name == "numba":
            return interval._interval_score_gufunc(
                observations[..., None], lower, upper, alpha
            )

        return interval._interval_score(
            observations[..., None], lower, upper, alpha, backend=backend
        )


def weighted_interval_score(
    observations: "ArrayLike",
    median: "Array",
    lower: "Array",
    upper: "Array",
    alpha: "Array",
    /,
    weight_median: Optional[float] = None,
    weight_alpha: Optional["Array"] = None,
    axis: int = -1,
    *,
    backend: "Backend" = None,
) -> "Array":
    r"""Compute the Interval Score or Winkler Score [(Bracher J, Ray EL, Gneiting T, Reich NG, 2022)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008618) for 1 - $\alpha$ prediction intervals PI = [lower, upper].

    The weighted interval score (WIS) is defined as

    $\text{WIS}_{\alpha_{0:K}}(F, y) = \frac{1}{K+0.5}(w_0 \times |y - m| + \sum_{k=1}^K (w_k \times IS_{\alpha_k}(F, y)))$

    where $m$ denotes the median prediction, $w_0$ denotes the weight of the median prediction,
    $IS_{\alpha_k}(F, y)$ denotes the interval score for the $1 - \alpha$ prediction interval and
    $w_k$ is the according weight. The WIS is calculated for a set of (central) PIs and the predictive
    median. The weights are an optional parameter and default weight
    is the canonical weight $w_k = \frac{2}{\alpha_k}$ and $w_0 = 0.5$. Using the canonical weights, the WIS
    can be used to approximate the CRPS.

    Parameters
    ----------
    observations: ArrayLike
        The observed values.
    median: Array
        The median prediction
    lower: Array
        The predicted lower bound of the prediction interval.
    upper: Array
        The predicted upper bound of the prediction interval.
    alpha: Array
        The 1 - alpha level for the prediction interval.
    weight_median: float
        The weight for the median prediction.
    weight_alpha: Array
        The weights for the PI.
    axis: int
        The axis corresponding to the ensemble. Default is the last axis.
    backend: str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    score: Array
        An array of interval scores for each observation, which should be averaged to get meaningful values.
    """
    if weight_alpha is None:
        weight_alpha = alpha / 2
    if weight_median is None:
        weight_median = 0.5

    B = backends.active if backend is None else backends[backend]
    observations, median, lower, upper, alpha, weight_alpha, weight_median = map(
        B.asarray,
        (observations, median, lower, upper, alpha, weight_alpha, weight_median),
    )

    if axis != -1:
        lower = B.moveaxis(lower, axis, -1)
        upper = B.moveaxis(upper, axis, -1)

    if B.name == "numba":
        return interval._weighted_interval_score_gufunc(
            observations, median, lower, upper, alpha, weight_median, weight_alpha
        )

    return interval._weighted_interval_score(
        observations,
        median,
        lower,
        upper,
        alpha,
        weight_median,
        weight_alpha,
        backend=backend,
    )
