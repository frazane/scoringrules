import typing as tp

from scoringrules.backend import backends
from scoringrules.core import kernels
from scoringrules.core.utils import multivariate_array_check

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike, Backend


def gksuv_ensemble(
    obs: "ArrayLike",
    fct: "Array",
    /,
    m_axis: int = -1,
    *,
    estimator: str = "nrg",
    backend: "Backend" = None,
) -> "Array":
    r"""Compute the univariate Gaussian Kernel Score (GKS) for a finite ensemble.

    The GKS is the kernel score associated with the Gaussian kernel

    .. math::
       k(x_{1}, x_{2}) = \exp \left(- \frac{(x_{1} - x_{2})^{2}}{2} \right).

    Given an ensemble forecast :math:`F_{ens}` comprised of members :math:`x_{1}, \dots, x_{M}`,
    the GKS is

    .. math::
      \text{GKS}(F_{ens}, y)= - \frac{1}{M} \sum_{m=1}^{M} k(x_{m}, y) + \frac{1}{2 M^{2}} \sum_{m=1}^{M} \sum_{j=1}^{M} k(x_{m}, x_{j}) + \frac{1}{2}k(y, y)

    If the fair estimator is to be used, then :math:`M^{2}` in the second component of the right-hand-side
    is replaced with :math:`M(M - 1)`.

    Parameters
    ----------
    obs : array_like
        The observed values.
    fct : array_like
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the last axis.
    m_axis : int
        The axis corresponding to the ensemble. Default is the last axis.
    estimator : str
        Indicates the estimator to be used.
    backend : str
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
    obs, fct = map(B.asarray, (obs, fct))

    if m_axis != -1:
        fct = B.moveaxis(fct, m_axis, -1)

    if backend == "numba":
        if estimator not in kernels.estimator_gufuncs:
            raise ValueError(
                f"{estimator} is not a valid estimator. "
                f"Must be one of {kernels.estimator_gufuncs.keys()}"
            )
        else:
            return kernels.estimator_gufuncs[estimator](obs, fct)

    return kernels.ensemble_uv(obs, fct, estimator, backend=backend)


def twgksuv_ensemble(
    obs: "ArrayLike",
    fct: "Array",
    /,
    a: float = float("-inf"),
    b: float = float("inf"),
    m_axis: int = -1,
    *,
    v_func: tp.Callable[["ArrayLike"], "ArrayLike"] = None,
    estimator: str = "nrg",
    backend: "Backend" = None,
) -> "Array":
    r"""Compute the Threshold-Weighted univariate Gaussian Kernel Score (GKS) for a finite ensemble.

    Computation is performed using the ensemble representation of threshold-weighted kernel scores in
    [Allen et al. (2022)](https://arxiv.org/abs/2202.12732).

    .. math::
       \mathrm{twGKS}(F_{ens}, y) = - \frac{1}{M} \sum_{m = 1}^{M} k(v(x_{m}), v(y)) + \frac{1}{2 M^{2}} \sum_{m = 1}^{M} \sum_{j = 1}^{M} k(v(x_{m}), v(x_{j})) + \frac{1}{2} k(v(y), v(y)),

    where :math:`F_{ens}(x) = \sum_{m=1}^{M} 1 \{ x_{m} \leq x \}/M` is the empirical
    distribution function associated with an ensemble forecast :math:`x_{1}, \dots, x_{M}` with
    :math:`M` members, :math:`v` is the chaining function used to target particular outcomes, and

    .. math::
       k(x_{1}, x_{2}) = \exp \left(- \frac{(x_{1} - x_{2})^{2}}{2} \right)

    is the Gaussian kernel.

    Parameters
    ----------
    obs : array_like
        The observed values.
    fct : array_like
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the last axis.
    a : float
        The lower bound to be used in the default weight function that restricts attention
        to values in the range [a, b].
    b : float
        The upper bound to be used in the default weight function that restricts attention
        to values in the range [a, b].
    m_axis : int
        The axis corresponding to the ensemble. Default is the last axis.
    v_func : callable, array_like -> array_like
        Chaining function used to emphasise particular outcomes. For example, a function that
        only considers values above a certain threshold :math:`t` by projecting forecasts and observations
        to :math:`[t, \inf)`.
    backend : str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    twgks : array_like
        The twGKS between the forecast ensemble and obs for the chosen chaining function.

    Examples
    --------
    >>> import numpy as np
    >>> import scoringrules as sr
    >>>
    >>> def v_func(x):
    >>>    return np.maximum(x, -1.0)
    >>>
    >>> sr.twgksuv_ensemble(obs, pred, v_func=v_func)
    """
    if v_func is None:
        B = backends.active if backend is None else backends[backend]
        a, b, obs, fct = map(B.asarray, (a, b, obs, fct))

        def v_func(x):
            return B.minimum(B.maximum(x, a), b)

    obs, fct = map(v_func, (obs, fct))
    return gksuv_ensemble(
        obs,
        fct,
        m_axis=m_axis,
        estimator=estimator,
        backend=backend,
    )


def owgksuv_ensemble(
    obs: "ArrayLike",
    fct: "Array",
    /,
    a: float = float("-inf"),
    b: float = float("inf"),
    m_axis: int = -1,
    *,
    w_func: tp.Callable[["ArrayLike"], "ArrayLike"] = None,
    backend: "Backend" = None,
) -> "Array":
    r"""Compute the univariate Outcome-Weighted Gaussian Kernel Score (owGKS) for a finite ensemble.

    Computation is performed using the ensemble representation of the owCRPS in
    [Allen et al. (2022)](https://arxiv.org/abs/2202.12732):

    .. math::
       \mathrm{owGKS}(F_{ens}, y) = -\frac{1}{M \bar{w}} \sum_{m = 1}^{M} k(x_{m}, y)w(x_{m})w(y) - \frac{1}{2 M^{2} \bar{w}^{2}} \sum_{m = 1}^{M} \sum_{j = 1}^{M} k(x_{m}, x_{j})w(x_{m})w(x_{j})w(y),

    where :math:`F_{ens}(x) = \sum_{m=1}^{M} 1\{ x_{m} \leq x \}/M` is the empirical
    distribution function associated with an ensemble forecast :math:`x_{1}, \dots, x_{M}` with
    :math:`M` members, :math:`w` is the chosen weight function, :math:`\bar{w} = \sum_{m=1}^{M}w(x_{m})/M`,
    and

    .. math::
       k(x_{1}, x_{2}) = \exp \left(- \frac{(x_{1} - x_{2})^{2}}{2} \right)

    is the Gaussian kernel.

    Parameters
    ----------
    obs : array_like
        The observed values.
    fct : array_like
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the last axis.
    a : float
        The lower bound to be used in the default weight function that restricts attention
        to values in the range [a, b].
    b : float
        The upper bound to be used in the default weight function that restricts attention
        to values in the range [a, b].
    m_axis : int
        The axis corresponding to the ensemble. Default is the last axis.
    w_func : callable, array_like -> array_like
        Weight function used to emphasise particular outcomes.
    backend : str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    score : array_like
        The owGKS between the forecast ensemble and obs for the chosen weight function.

    Examples
    --------
    >>> import numpy as np
    >>> import scoringrules as sr
    >>>
    >>> def w_func(x):
    >>>    return (x > -1).astype(float)
    >>>
    >>> sr.owgksuv_ensemble(obs, pred, w_func=w_func)
    """
    B = backends.active if backend is None else backends[backend]
    obs, fct = map(B.asarray, (obs, fct))

    if m_axis != -1:
        fct = B.moveaxis(fct, m_axis, -1)

    if w_func is None:

        def w_func(x):
            return ((a <= x) & (x <= b)) * 1.0

    obs_weights, fct_weights = map(w_func, (obs, fct))
    obs_weights, fct_weights = map(B.asarray, (obs_weights, fct_weights))

    if backend == "numba":
        return kernels.estimator_gufuncs["ow"](obs, fct, obs_weights, fct_weights)

    return kernels.ow_ensemble_uv(obs, fct, obs_weights, fct_weights, backend=backend)


def vrgksuv_ensemble(
    obs: "ArrayLike",
    fct: "Array",
    /,
    a: float = float("-inf"),
    b: float = float("inf"),
    m_axis: int = -1,
    *,
    w_func: tp.Callable[["ArrayLike"], "ArrayLike"] = None,
    backend: "Backend" = None,
) -> "Array":
    r"""Estimate the Vertically Re-scaled Gaussian Kernel Score (vrGKS) for a finite ensemble.

    Computation is performed using the ensemble representation of vertically re-scaled kernel scores in
    [Allen et al. (2022)](https://arxiv.org/abs/2202.12732):

    .. math::
       \mathrm{vrGKS}(F_{ens}, y) = - \frac{1}{M} \sum_{m = 1}^{M} k(x_{m}, y)w(x_{m})w(y) + \frac{1}{2 M^{2}} \sum_{m = 1}^{M} \sum_{j = 1}^{M} k(x_{m}, x_{j})w(x_{m})w(x_{j}) + \frac{1}{2} k(y, y)w(y)w(y),

    where :math:`F_{ens}(x) = \sum_{m=1}^{M} 1 \{ x_{m} \leq x \}/M` is the empirical
    distribution function associated with an ensemble forecast :math:`x_{1}, \dots, x_{M}` with
    :math:`M` members, :math:`w` is the chosen weight function, :math:`\bar{w} = \sum_{m=1}^{M}w(x_{m})/M`, and

    .. math::
       k(x_{1}, x_{2}) = \exp \left(- \frac{(x_{1} - x_{2})^{2}}{2} \right)

    is the Gaussian kernel.

    Parameters
    ----------
    obs : array_like
        The observed values.
    fct : array_like
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the last axis.
    a : float
        The lower bound to be used in the default weight function that restricts attention
        to values in the range [a, b].
    b : float
        The upper bound to be used in the default weight function that restricts attention
        to values in the range [a, b].
    m_axis : int
        The axis corresponding to the ensemble. Default is the last axis.
    w_func : callable, array_like -> array_like
        Weight function used to emphasise particular outcomes.
    backend : str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    score : array_like
        The vrGKS between the forecast ensemble and obs for the chosen weight function.

    Examples
    --------
    >>> import numpy as np
    >>> import scoringrules as sr
    >>>
    >>> def w_func(x):
    >>>    return (x > -1).astype(float)
    >>>
    >>> sr.vrgksuv_ensemble(obs, pred, w_func=w_func)
    """
    B = backends.active if backend is None else backends[backend]
    obs, fct = map(B.asarray, (obs, fct))

    if m_axis != -1:
        fct = B.moveaxis(fct, m_axis, -1)

    if w_func is None:

        def w_func(x):
            return ((a <= x) & (x <= b)) * 1.0

    obs_weights, fct_weights = map(w_func, (obs, fct))
    obs_weights, fct_weights = map(B.asarray, (obs_weights, fct_weights))

    if backend == "numba":
        return kernels.estimator_gufuncs["vr"](obs, fct, obs_weights, fct_weights)

    return kernels.vr_ensemble_uv(obs, fct, obs_weights, fct_weights, backend=backend)


def gksmv_ensemble(
    obs: "Array",
    fct: "Array",
    /,
    m_axis: int = -2,
    v_axis: int = -1,
    *,
    estimator: str = "nrg",
    backend: "Backend" = None,
) -> "Array":
    r"""Compute the multivariate Gaussian Kernel Score (GKS) for a finite ensemble.

    The GKS is the kernel score associated with the Gaussian kernel

    .. math::
       k(x_{1}, x_{2}) = \exp \left(- \frac{ \| x_{1} - x_{2} \| ^{2}}{2} \right),

    where :math:` \| \cdot \|` is the euclidean norm.

    Given an ensemble forecast :math:`F_{ens}` comprised of multivariate members :math:`\mathbf{x}_{1}, \dots, \mathbf{x}_{M}`,
    the GKS is

    .. math::
      \text{GKS}(F_{ens}, y)= - \frac{1}{M} \sum_{m=1}^{M} k(\mathbf{x}_{m}, \mathbf{y}) + \frac{1}{2 M^{2}} \sum_{m=1}^{M} \sum_{j=1}^{M} k(\mathbf{x}_{m}, \mathbf{x}_{j}) + \frac{1}{2}k(y, y)

    If the fair estimator is to be used, then :math:`M^{2}` in the second component of the right-hand-side
    is replaced with :math:`M(M - 1)`.


    Parameters
    ----------
    obs : array_like
        The observed values, where the variables dimension is by default the last axis.
    fct : array_like
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the second last axis and the variables dimension by the last axis.
    m_axis : int
        The axis corresponding to the ensemble dimension on the forecasts array. Defaults to -2.
    v_axis : int
        The axis corresponding to the variables dimension on the forecasts array (or the observations
        array with an extra dimension on `m_axis`). Defaults to -1.
    estimator : str
        Indicates the estimator to be used.
    backend : str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    score : array_like
        The GKS between the forecast ensemble and obs.
    """
    backend = backend if backend is not None else backends._active
    obs, fct = multivariate_array_check(obs, fct, m_axis, v_axis, backend=backend)

    if backend == "numba":
        if estimator not in kernels.estimator_gufuncs_mv:
            raise ValueError(
                f"{estimator} is not a valid estimator. "
                f"Must be one of {kernels.estimator_gufuncs_mv.keys()}"
            )
        else:
            return kernels.estimator_gufuncs_mv[estimator](obs, fct)

    return kernels.ensemble_mv(obs, fct, estimator, backend=backend)


def twgksmv_ensemble(
    obs: "Array",
    fct: "Array",
    v_func: tp.Callable[["ArrayLike"], "ArrayLike"],
    /,
    m_axis: int = -2,
    v_axis: int = -1,
    *,
    backend: "Backend" = None,
) -> "Array":
    r"""Compute the Threshold-Weighted Gaussian Kernel Score (twGKS) for a finite multivariate ensemble.

    Computation is performed using the ensemble representation of threshold-weighted kernel scores in
    [Allen et al. (2022)](https://arxiv.org/abs/2202.12732):

    .. math::
       \mathrm{twGKS}(F_{ens}, y) = - \frac{1}{M} \sum_{m = 1}^{M} k(v(x_{m}), v(y)) + \frac{1}{2 M^{2}} \sum_{m = 1}^{M} \sum_{j = 1}^{M} k(v(x_{m}), v(x_{j})) + \frac{1}{2} k(v(y), v(y)),

    where :math:`F_{ens}` is the ensemble forecast :math:`\mathbf{x}_{1}, \dots, \mathbf{x}_{M}` with
    :math:`M` members, :math:`\| \cdotp \|` is the Euclidean distance, :math:`v` is the chaining function
    used to target particular outcomes, and

    .. math::
       k(x_{1}, x_{2}) = \exp \left(- \frac{(x_{1} - x_{2})^{2}}{2} \right)

    is the Gaussian kernel.

    Parameters
    ----------
    obs : array_like
        The observed values, where the variables dimension is by default the last axis.
    fct : array_like
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the second last axis and the variables dimension by the last axis.
    v_func : callable, array_like -> array_like
        Chaining function used to emphasise particular outcomes.
    m_axis : int
        The axis corresponding to the ensemble dimension. Defaults to -2.
    v_axis : int or tuple of ints
        The axis corresponding to the variables dimension. Defaults to -1.
    backend : str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    score : array_like
        The computed Threshold-Weighted Gaussian Kernel Score.
    """
    obs, fct = map(v_func, (obs, fct))
    return gksmv_ensemble(obs, fct, m_axis=m_axis, v_axis=v_axis, backend=backend)


def owgksmv_ensemble(
    obs: "Array",
    fct: "Array",
    w_func: tp.Callable[["ArrayLike"], "ArrayLike"],
    /,
    m_axis: int = -2,
    v_axis: int = -1,
    *,
    backend: "Backend" = None,
) -> "Array":
    r"""Compute the multivariate Outcome-Weighted Gaussian Kernel Score (owGKS) for a finite ensemble.



    Given an ensemble forecast :math:`F_{ens}` comprised of multivariate members :math:`\mathbf{x}_{1}, \dots, \mathbf{x}_{M}`,
    the GKS is

    .. math::
        \text{GKS}(F_{ens}, y)= - \frac{1}{M} \sum_{m=1}^{M} k(\mathbf{x}_{m}, \mathbf{y})
        + \frac{1}{2 M^{2}} \sum_{m=1}^{M} \sum_{j=1}^{M} k(\mathbf{x}_{m}, \mathbf{x}_{j}) + \frac{1}{2}k(y, y)

    If the fair estimator is to be used, then :math:`M^{2}` in the second component of the right-hand-side
    is replaced with :math:`M(M - 1)`.

    Computation is performed using the ensemble representation of outcome-weighted kernel scores in
    [Allen et al. (2022)](https://arxiv.org/abs/2202.12732):

    .. math::
        \mathrm{owGKS}(F_{ens}, \mathbf{y}) = - \frac{1}{M \bar{w}} \sum_{m = 1}^{M} k(\mathbf{x}_{m}, \mathbf{y}) w(\mathbf{x}_{m}) w(\mathbf{y})
        + \frac{1}{2 M^{2} \bar{w}^{2}} \sum_{m = 1}^{M} \sum_{j = 1}^{M} k(\mathbf{x}_{m}, \mathbf{x}_{j}) w(\mathbf{x}_{m}) w(\mathbf{x}_{j}) w(\mathbf{y})
        + \frac{1}{2}k(\mathbf{y}, \mathbf{y})w(\mathbf{y}),

    where :math:`F_{ens}` is the ensemble forecast :math:`\mathbf{x}_{1}, \dots, \mathbf{x}_{M}` with
    :math:`M` members, :math:`\| \cdotp \|` is the Euclidean distance, :math:`w` is the chosen weight function,
    :math:`\bar{w} = \sum_{m=1}^{M}w(\mathbf{x}_{m})/M`, and

    .. math::
        k(x_{1}, x_{2}) = \exp \left(- \frac{ \| x_{1} - x_{2} \| ^{2}}{2} \right),

    is the multivariate Gaussian kernel, with :math:` \| \cdot \|` the Euclidean norm.


    Parameters
    ----------
    obs : array_like
        The observed values, where the variables dimension is by default the last axis.
    fct : array_like
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the second last axis and the variables dimension by the last axis.
    w_func : callable, array_like -> array_like
        Weight function used to emphasise particular outcomes.
    m_axis : int
        The axis corresponding to the ensemble dimension. Defaults to -2.
    v_axis : int or tuple of ints
        The axis corresponding to the variables dimension. Defaults to -1.
    backend : str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    score : array_like
        The computed Outcome-Weighted GKS.
    """
    B = backends.active if backend is None else backends[backend]

    obs, fct = multivariate_array_check(obs, fct, m_axis, v_axis, backend=backend)

    fct_weights = B.apply_along_axis(w_func, fct, -1)
    obs_weights = B.apply_along_axis(w_func, obs, -1)

    if B.name == "numba":
        return kernels.estimator_gufuncs_mv["ow"](obs, fct, obs_weights, fct_weights)

    return kernels.ow_ensemble_mv(obs, fct, obs_weights, fct_weights, backend=backend)


def vrgksmv_ensemble(
    obs: "Array",
    fct: "Array",
    w_func: tp.Callable[["ArrayLike"], "ArrayLike"],
    /,
    *,
    m_axis: int = -2,
    v_axis: int = -1,
    backend: "Backend" = None,
) -> "Array":
    r"""Compute the Vertically Re-scaled Gaussian Kernel Score (vrGKS) for a finite multivariate ensemble.

    Computation is performed using the ensemble representation of vertically re-scaled kernel scores  in
    [Allen et al. (2022)](https://arxiv.org/abs/2202.12732):

    .. math::
        \mathrm{vrGKS}(F_{ens}, \mathbf{y}) = & - \frac{1}{M} \sum_{m = 1}^{M} k(\mathbf{x}_{m}, \mathbf{y}) w(\mathbf{x}_{m}) w(\mathbf{y})
        + \frac{1}{2 M^{2}} \sum_{m = 1}^{M} \sum_{j = 1}^{M} k(\mathbf{x}_{m}, \mathbf{x}_{j}) w(\mathbf{x}_{m}) w(\mathbf{x_{j}}) + \frac{1}{2} k(y, y)w(y)w(y),

    where :math:`F_{ens}` is the ensemble forecast :math:`\mathbf{x}_{1}, \dots, \mathbf{x}_{M}` with
    :math:`M` members, :math:`w` is the weight function used to target particular outcomes, and

    .. math::
       k(x_{1}, x_{2}) = \exp \left(- \frac{ \| x_{1} - x_{2} \| ^{2}}{2} \right),

    is the multivariate Gaussian kernel, with :math:` \| \cdot \|` the Euclidean norm.


    Parameters
    ----------
    obs : array_like
        The observed values, where the variables dimension is by default the last axis.
    fct : array_like
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the second last axis and the variables dimension by the last axis.
    w_func : callable, array_like -> array_like
        Weight function used to emphasise particular outcomes.
    m_axis : int
        The axis corresponding to the ensemble dimension. Defaults to -2.
    v_axis : int or tuple of ints
        The axis corresponding to the variables dimension. Defaults to -1.
    backend: str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    score : array_like
        The computed Vertically Re-scaled Gaussian Kernel Score.
    """
    B = backends.active if backend is None else backends[backend]

    obs, fct = multivariate_array_check(obs, fct, m_axis, v_axis, backend=backend)

    fct_weights = B.apply_along_axis(w_func, fct, -1)
    obs_weights = B.apply_along_axis(w_func, obs, -1)

    if B.name == "numba":
        return kernels.estimator_gufuncs_mv["vr"](obs, fct, obs_weights, fct_weights)

    return kernels.vr_ensemble_mv(obs, fct, obs_weights, fct_weights, backend=backend)


__all__ = [
    "gksuv_ensemble",
    "twgksuv_ensemble",
    "owgksuv_ensemble",
    "vrgksuv_ensemble",
    "gksmv_ensemble",
    "twgksmv_ensemble",
    "owgksmv_ensemble",
    "vrgksmv_ensemble",
]
