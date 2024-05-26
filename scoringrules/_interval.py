import typing as tp

from scoringrules.backend import backends
from scoringrules.core import interval

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike, Backend


def interval_score(
    observations: "ArrayLike",
    lower: "Array",
    upper: "Array",
    alpha: "Array",
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

    Parameters
    ----------
    observations: ArrayLike
        The observed values.
    lower: Array
        The predicted lower bound of the prediction interval.
    upper: Array
        The predicted upper bound of the prediction interval.
    alpha: Array
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
    observations, lower, upper, alpha = map(
        B.asarray, (observations, lower, upper, alpha)
    )

    if axis != -1:
        lower = B.moveaxis(lower, axis, -1)
        upper = B.moveaxis(upper, axis, -1)

    if B.name == "numba":
        return interval._interval_score_gufunc(observations, lower, upper, alpha)

    return interval._interval_score(observations, lower, upper, alpha, backend=backend)
