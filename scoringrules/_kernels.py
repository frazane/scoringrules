import typing as tp

from scoringrules.backend import backends
from scoringrules.core import kernels
from scoringrules.core.utils import multivariate_array_check

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike, Backend


def gksuv_ensemble(
    observations: "ArrayLike",
    forecasts: "Array",
    /,
    axis: int = -1,
    *,
    estimator: str = "nrg",
    backend: "Backend" = None,
) -> "Array":
    r"""Estimate the univariate Gaussian Kernel Score (GKS) for a finite ensemble.

    The GKS is the kernel score associated with the Gaussian kernel

    $$ k(x_{1}, x_{2}) = \exp \left(- \frac{(x_{1} - x_{2})^{2}}{2} \right). $$

    Given an ensemble forecast $F_{ens}$ comprised of members $x_{1}, \dots, x_{M}$,
    the GKS is

    $$\text{GKS}(F_{ens}, y)= - \frac{1}{M} \sum_{m=1}^{M} k(x_{m}, y) + \frac{1}{2 M^{2}} \sum_{m=1}^{M} \sum_{j=1}^{M} k(x_{m}, x_{j}) + \frac{1}{2}k(y, y) $$

    If the fair estimator is to be used, then $M^{2}$ in the second component of the right-hand-side
    is replaced with $M(M - 1)$.

    Parameters
    ----------
    observations: ArrayLike
        The observed values.
    forecasts: ArrayLike
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the last axis.
    axis: int
        The axis corresponding to the ensemble. Default is the last axis.
    estimator: str
        Indicates the estimator to be used.
    backend: str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    score:
        The GKS between the forecast ensemble and obs.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.gks_ensemble(obs, pred)
    """
    B = backends.active if backend is None else backends[backend]
    observations, forecasts = map(B.asarray, (observations, forecasts))

    if estimator not in kernels.estimator_gufuncs:
        raise ValueError(
            f"{estimator} is not a valid estimator. "
            f"Must be one of {kernels.estimator_gufuncs.keys()}"
        )

    if axis != -1:
        forecasts = B.moveaxis(forecasts, axis, -1)

    if backend == "numba":
        return kernels.estimator_gufuncs[estimator](observations, forecasts)

    return kernels.ensemble_uv(observations, forecasts, estimator, backend=backend)


def twgksuv_ensemble(
    observations: "ArrayLike",
    forecasts: "Array",
    v_func: tp.Callable[["ArrayLike"], "ArrayLike"],
    /,
    axis: int = -1,
    *,
    estimator: str = "pwm",
    sorted_ensemble: bool = False,
    backend: "Backend" = None,
) -> "Array":
    r"""Estimate the Threshold-Weighted univariate Gaussian Kernel Score (GKS) for a finite ensemble.

    Computation is performed using the ensemble representation of threshold-weighted kernel scores in
    [Allen et al. (2022)](https://arxiv.org/abs/2202.12732).

    $$ \mathrm{twGKS}(F_{ens}, y) = - \frac{1}{M} \sum_{m = 1}^{M} k(v(x_{m}), v(y)) + \frac{1}{2 M^{2}} \sum_{m = 1}^{M} \sum_{j = 1}^{M} k(v(x_{m}), v(x_{j})) + \frac{1}{2} k(v(y), v(y)),$$

    where $F_{ens}(x) = \sum_{m=1}^{M} 1 \{ x_{m} \leq x \}/M$ is the empirical
    distribution function associated with an ensemble forecast $x_{1}, \dots, x_{M}$ with
    $M$ members, $v$ is the chaining function used to target particular outcomes, and

    $$ k(x_{1}, x_{2}) = \exp \left(- \frac{(x_{1} - x_{2})^{2}}{2} \right) $$

    is the Gaussian kernel.

    Parameters
    ----------
    observations: ArrayLike
        The observed values.
    forecasts: ArrayLike
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the last axis.
    v_func: tp.Callable
        Chaining function used to emphasise particular outcomes. For example, a function that
        only considers values above a certain threshold $t$ by projecting forecasts and observations
        to $[t, \inf)$.
    axis: int
        The axis corresponding to the ensemble. Default is the last axis.
    backend: str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    twgks: ArrayLike
        The twGKS between the forecast ensemble and obs for the chosen chaining function.

    Examples
    --------
    >>> import numpy as np
    >>> import scoringrules as sr
    >>>
    >>> def v_func(x):
    >>>    return np.maximum(x, -1.0)
    >>>
    >>> sr.twgksuv_ensemble(obs, pred, v_func)
    """
    observations, forecasts = map(v_func, (observations, forecasts))
    return gksuv_ensemble(
        observations,
        forecasts,
        axis=axis,
        sorted_ensemble=sorted_ensemble,
        estimator=estimator,
        backend=backend,
    )


def gksmv_ensemble(
    observations: "Array",
    forecasts: "Array",
    /,
    m_axis: int = -2,
    v_axis: int = -1,
    *,
    estimator: str = "nrg",
    backend: "Backend" = None,
) -> "Array":
    r"""Estimate the multivariate Gaussian Kernel Score (GKS) for a finite ensemble.

    The GKS is the kernel score associated with the Gaussian kernel

    $$ k(x_{1}, x_{2}) = \exp \left(- \frac{ \| x_{1} - x_{2} \| ^{2}}{2} \right), $$

    where $ \| \cdot \|$ is the euclidean norm.

    Given an ensemble forecast $F_{ens}$ comprised of multivariate members $\mathbf{x}_{1}, \dots, \mathbf{x}_{M}$,
    the GKS is

    $$\text{GKS}(F_{ens}, y)= - \frac{1}{M} \sum_{m=1}^{M} k(\mathbf{x}_{m}, \mathbf{y}) + \frac{1}{2 M^{2}} \sum_{m=1}^{M} \sum_{j=1}^{M} k(\mathbf{x}_{m}, \mathbf{x}_{j}) + \frac{1}{2}k(y, y) $$

    If the fair estimator is to be used, then $M^{2}$ in the second component of the right-hand-side
    is replaced with $M(M - 1)$.


    Parameters
    ----------
    observations: Array
        The observed values, where the variables dimension is by default the last axis.
    forecasts: Array
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the second last axis and the variables dimension by the last axis.
    m_axis: int
        The axis corresponding to the ensemble dimension on the forecasts array. Defaults to -2.
    v_axis: int
        The axis corresponding to the variables dimension on the forecasts array (or the observations
        array with an extra dimension on `m_axis`). Defaults to -1.
    estimator: str
        Indicates the estimator to be used.
    backend: str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    score:
        The GKS between the forecast ensemble and obs.
    """
    backend = backend if backend is not None else backends._active
    observations, forecasts = multivariate_array_check(
        observations, forecasts, m_axis, v_axis, backend=backend
    )

    if estimator not in kernels.estimator_gufuncs_mv:
        raise ValueError(
            f"{estimator} is not a valid estimator. "
            f"Must be one of {kernels.estimator_gufuncs_mv.keys()}"
        )

    if backend == "numba":
        return kernels.estimator_gufuncs_mv[estimator](observations, forecasts)

    return kernels.ensemble_mv(observations, forecasts, estimator, backend=backend)
