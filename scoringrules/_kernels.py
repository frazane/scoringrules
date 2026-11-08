import typing as tp

from scoringrules._dispatch import (
    gufunc_inputs_supported,
    resolve_backend_arg,
    use_numba,
)
from scoringrules.backend import get_namespace
from scoringrules.core import kernels
from scoringrules.core.utils import (
    asarray_matching,
    estimator_check,
    multivariate_array_check,
    multivariate_weight_check,
    mv_weighted_score_chain,
    mv_weighted_score_weights,
    univariate_array_check,
    univariate_weight_check,
    uv_outcome_weights,
    uv_weighted_score_chain,
)

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike, Backend


_KS_ESTIMATORS = ("nrg", "fair", "akr", "akr_circperm")


def gksuv_ensemble(
    obs: "ArrayLike",
    fct: "Array",
    m_axis: int = -1,
    *,
    ens_w: "Array" = None,
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
    ens_w : array
        Weights assigned to the ensemble members. Array with the same shape as fct.
        Default is equal weighting. Weights are normalised so that they sum to one across the ensemble members.
    estimator : str
        Indicates the estimator to be used.
    backend : str, optional
        Deprecated, removed in 1.0. The array framework is inferred from the inputs;
        use ``scoringrules.set_config`` or ``scoringrules.config_context`` to control
        numba.

    Returns
    -------
    score:
        The GKS between the forecast ensemble and obs.

    Examples
    --------
    >>> import numpy as np
    >>> import scoringrules as sr
    >>> rng = np.random.default_rng(123)
    >>> obs = rng.normal(size=3)
    >>> fct = rng.normal(size=(3, 10))
    >>> sr.gksuv_ensemble(obs, fct)
    array([0.30799922, 0.17469789, 0.17174067])
    """
    resolve_backend_arg(backend)
    numba = use_numba(("gksuv_ensemble", estimator), backend, obs, fct, aux=(ens_w,))
    xp = get_namespace(obs, fct)
    obs, fct = univariate_array_check(obs, fct, m_axis, xp=xp)
    if ens_w is not None:
        ens_w = univariate_weight_check(ens_w, fct, m_axis, xp=xp)

    if numba:
        estimator_check(estimator, _KS_ESTIMATORS)
        if ens_w is None:
            return kernels.estimator_gufuncs_uv[estimator](obs, fct)
        return kernels.estimator_gufuncs_uv_w[estimator](obs, fct, ens_w)
    if ens_w is None:
        return kernels.ensemble_uv(obs, fct, estimator=estimator, xp=xp)
    return kernels.ensemble_uv_w(obs, fct, ens_w, estimator=estimator, xp=xp)


def twgksuv_ensemble(
    obs: "ArrayLike",
    fct: "Array",
    a: float = float("-inf"),
    b: float = float("inf"),
    m_axis: int = -1,
    *,
    ens_w: "Array" = None,
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
    ens_w : array
        Weights assigned to the ensemble members. Array with the same shape as fct.
        Default is equal weighting. Weights are normalised so that they sum to one across the ensemble members.
    v_func : callable, array_like -> array_like
        Chaining function used to emphasise particular outcomes. For example, a function that
        only considers values above a certain threshold :math:`t` by projecting forecasts and observations
        to :math:`[t, \inf)`.
    backend : str, optional
        Deprecated, removed in 1.0. The array framework is inferred from the inputs;
        use ``scoringrules.set_config`` or ``scoringrules.config_context`` to control
        numba.

    Returns
    -------
    twgks : array_like
        The twGKS between the forecast ensemble and obs for the chosen chaining function.

    Examples
    --------
    >>> import numpy as np
    >>> import scoringrules as sr
    >>> rng = np.random.default_rng(123)
    >>> obs = rng.normal(size=3)
    >>> fct = rng.normal(size=(3, 10))
    >>> def v_func(x):
    ...     return np.maximum(x, -1.0)
    >>> sr.twgksuv_ensemble(obs, fct, v_func=v_func)
    array([0.31423505, 0.14243174, 0.17610499])
    """
    # warns here, at the user's call; the score called below does not
    backend = resolve_backend_arg(backend)
    xp = get_namespace(obs, fct)
    # the bounds follow the forecast precision (a float64 0-d array would upcast
    # float32 numpy inputs, a float32 tensor would round them for float64 torch)
    a, b = asarray_matching(a, b, obs, fct, xp=xp)[:2]
    obs, fct = uv_weighted_score_chain(obs, fct, a=a, b=b, v_func=v_func, xp=xp)
    return gksuv_ensemble(
        obs,
        fct,
        m_axis=m_axis,
        ens_w=ens_w,
        estimator=estimator,
        backend=backend,
    )


def owgksuv_ensemble(
    obs: "ArrayLike",
    fct: "Array",
    a: float = float("-inf"),
    b: float = float("inf"),
    m_axis: int = -1,
    *,
    ens_w: "Array" = None,
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
    ens_w : array
        Weights assigned to the ensemble members. Array with the same shape as fct.
        Default is equal weighting. Weights are normalised so that they sum to one across the ensemble members.
    w_func : callable, array_like -> array_like
        Weight function used to emphasise particular outcomes.
    backend : str, optional
        Deprecated, removed in 1.0. The array framework is inferred from the inputs;
        use ``scoringrules.set_config`` or ``scoringrules.config_context`` to control
        numba.

    Returns
    -------
    score : array_like
        The owGKS between the forecast ensemble and obs for the chosen weight function.

    Examples
    --------
    >>> import numpy as np
    >>> import scoringrules as sr
    >>> rng = np.random.default_rng(123)
    >>> obs = rng.normal(size=3)
    >>> fct = rng.normal(size=(3, 10))
    >>> def w_func(x):
    ...     return (x > -1).astype(float)
    >>> sr.owgksuv_ensemble(obs, fct, w_func=w_func)
    array([0.38847902, 0.17524147, 0.14577796])
    """
    resolve_backend_arg(backend)
    numba = use_numba("owgksuv_ensemble", backend, obs, fct, aux=(ens_w,))
    xp = get_namespace(obs, fct)
    obs, fct = univariate_array_check(obs, fct, m_axis, xp=xp)
    obs_w, fct_w = uv_outcome_weights(obs, fct, a, b, w_func, xp=xp)
    numba = numba and gufunc_inputs_supported(obs_w, fct_w)  # e.g. masked w_func output
    if ens_w is not None:
        ens_w = univariate_weight_check(ens_w, fct, m_axis, xp=xp)

    if numba:
        if ens_w is None:
            return kernels.estimator_gufuncs_uv["ow"](obs, fct, obs_w, fct_w)
        return kernels.estimator_gufuncs_uv_w["ow"](obs, fct, obs_w, fct_w, ens_w)
    if ens_w is None:
        return kernels.ow_ensemble_uv(obs, fct, obs_w, fct_w, xp=xp)
    return kernels.ow_ensemble_uv_w(obs, fct, obs_w, fct_w, ens_w, xp=xp)


def vrgksuv_ensemble(
    obs: "ArrayLike",
    fct: "Array",
    a: float = float("-inf"),
    b: float = float("inf"),
    m_axis: int = -1,
    *,
    ens_w: "Array" = None,
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
    ens_w : array
        Weights assigned to the ensemble members. Array with the same shape as fct.
        Default is equal weighting. Weights are normalised so that they sum to one across the ensemble members.
    w_func : callable, array_like -> array_like
        Weight function used to emphasise particular outcomes.
    backend : str, optional
        Deprecated, removed in 1.0. The array framework is inferred from the inputs;
        use ``scoringrules.set_config`` or ``scoringrules.config_context`` to control
        numba.

    Returns
    -------
    score : array_like
        The vrGKS between the forecast ensemble and obs for the chosen weight function.

    Examples
    --------
    >>> import numpy as np
    >>> import scoringrules as sr
    >>> rng = np.random.default_rng(123)
    >>> obs = rng.normal(size=3)
    >>> fct = rng.normal(size=(3, 10))
    >>> def w_func(x):
    ...     return (x > -1).astype(float)
    >>> sr.vrgksuv_ensemble(obs, fct, w_func=w_func)
    array([0.3652057 , 0.17914915, 0.14754764])
    """
    resolve_backend_arg(backend)
    numba = use_numba("vrgksuv_ensemble", backend, obs, fct, aux=(ens_w,))
    xp = get_namespace(obs, fct)
    obs, fct = univariate_array_check(obs, fct, m_axis, xp=xp)
    obs_w, fct_w = uv_outcome_weights(obs, fct, a, b, w_func, xp=xp)
    numba = numba and gufunc_inputs_supported(obs_w, fct_w)  # e.g. masked w_func output
    if ens_w is not None:
        ens_w = univariate_weight_check(ens_w, fct, m_axis, xp=xp)

    if numba:
        if ens_w is None:
            return kernels.estimator_gufuncs_uv["vr"](obs, fct, obs_w, fct_w)
        return kernels.estimator_gufuncs_uv_w["vr"](obs, fct, obs_w, fct_w, ens_w)
    if ens_w is None:
        return kernels.vr_ensemble_uv(obs, fct, obs_w, fct_w, xp=xp)
    return kernels.vr_ensemble_uv_w(obs, fct, obs_w, fct_w, ens_w, xp=xp)


def gksmv_ensemble(
    obs: "Array",
    fct: "Array",
    m_axis: int = -2,
    v_axis: int = -1,
    *,
    ens_w: "Array" = None,
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
    ens_w : array_like
        Weights assigned to the ensemble members. Array with one less dimension than fct (without the v_axis dimension).
        Default is equal weighting. Weights are normalised so that they sum to one across the ensemble members.
    estimator : str
        Indicates the estimator to be used.
    backend : str, optional
        Deprecated, removed in 1.0. The array framework is inferred from the inputs;
        use ``scoringrules.set_config`` or ``scoringrules.config_context`` to control
        numba.

    Returns
    -------
    score : array_like
        The GKS between the forecast ensemble and obs.
    """
    resolve_backend_arg(backend)
    numba = use_numba(("gksmv_ensemble", estimator), backend, obs, fct, aux=(ens_w,))
    xp = get_namespace(obs, fct)
    obs, fct = multivariate_array_check(obs, fct, m_axis, v_axis, xp=xp)
    if ens_w is not None:
        ens_w = multivariate_weight_check(ens_w, fct, m_axis, v_axis, xp=xp)

    if numba:
        estimator_check(estimator, _KS_ESTIMATORS)
        if ens_w is None:
            return kernels.estimator_gufuncs_mv[estimator](obs, fct)
        return kernels.estimator_gufuncs_mv_w[estimator](obs, fct, ens_w)
    if ens_w is None:
        return kernels.ensemble_mv(obs, fct, estimator=estimator, xp=xp)
    return kernels.ensemble_mv_w(obs, fct, ens_w, estimator=estimator, xp=xp)


def twgksmv_ensemble(
    obs: "Array",
    fct: "Array",
    v_func: tp.Callable[["ArrayLike"], "ArrayLike"],
    m_axis: int = -2,
    v_axis: int = -1,
    *,
    ens_w: "Array" = None,
    estimator: str = "nrg",
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
    ens_w : array_like
        Weights assigned to the ensemble members. Array with one less dimension than fct (without the v_axis dimension).
        Default is equal weighting. Weights are normalised so that they sum to one across the ensemble members.
    estimator : str
        Indicates the estimator to be used.
    backend : str, optional
        Deprecated, removed in 1.0. The array framework is inferred from the inputs;
        use ``scoringrules.set_config`` or ``scoringrules.config_context`` to control
        numba.

    Returns
    -------
    score : array_like
        The computed Threshold-Weighted Gaussian Kernel Score.
    """
    # warns here, at the user's call; the score called below does not
    backend = resolve_backend_arg(backend)
    obs, fct = mv_weighted_score_chain(obs, fct, v_func)
    return gksmv_ensemble(
        obs,
        fct,
        m_axis=m_axis,
        v_axis=v_axis,
        ens_w=ens_w,
        estimator=estimator,
        backend=backend,
    )


def owgksmv_ensemble(
    obs: "Array",
    fct: "Array",
    w_func: tp.Callable[["ArrayLike"], "ArrayLike"],
    m_axis: int = -2,
    v_axis: int = -1,
    *,
    ens_w: "Array" = None,
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
    ens_w : array_like
        Weights assigned to the ensemble members. Array with one less dimension than fct (without the v_axis dimension).
        Default is equal weighting. Weights are normalised so that they sum to one across the ensemble members.
    backend : str, optional
        Deprecated, removed in 1.0. The array framework is inferred from the inputs;
        use ``scoringrules.set_config`` or ``scoringrules.config_context`` to control
        numba.

    Returns
    -------
    score : array_like
        The computed Outcome-Weighted GKS.
    """
    resolve_backend_arg(backend)
    numba = use_numba("owgksmv_ensemble", backend, obs, fct, aux=(ens_w,))
    xp = get_namespace(obs, fct)
    obs, fct = multivariate_array_check(obs, fct, m_axis, v_axis, xp=xp)
    obs_w, fct_w = mv_weighted_score_weights(obs, fct, w_func, xp=xp)
    numba = numba and gufunc_inputs_supported(obs_w, fct_w)  # e.g. masked w_func output
    if ens_w is not None:
        ens_w = multivariate_weight_check(ens_w, fct, m_axis, v_axis, xp=xp)

    if numba:
        if ens_w is None:
            return kernels.estimator_gufuncs_mv["ow"](obs, fct, obs_w, fct_w)
        return kernels.estimator_gufuncs_mv_w["ow"](obs, fct, obs_w, fct_w, ens_w)
    if ens_w is None:
        return kernels.ow_ensemble_mv(obs, fct, obs_w, fct_w, xp=xp)
    return kernels.ow_ensemble_mv_w(obs, fct, obs_w, fct_w, ens_w, xp=xp)


def vrgksmv_ensemble(
    obs: "Array",
    fct: "Array",
    w_func: tp.Callable[["ArrayLike"], "ArrayLike"],
    m_axis: int = -2,
    v_axis: int = -1,
    *,
    ens_w: "Array" = None,
    backend: "Backend" = None,
) -> "Array":
    r"""Compute the Vertically Re-scaled Gaussian Kernel Score (vrGKS) for a finite multivariate ensemble.

    Computation is performed using the ensemble representation of vertically re-scaled kernel scores  in
    [Allen et al. (2022)](https://arxiv.org/abs/2202.12732):

    .. math::
        \begin{aligned}
            \mathrm{vrGKS}(F_{ens}, \mathbf{y}) = & - \frac{1}{M} \sum_{m = 1}^{M} k(\mathbf{x}_{m}, \mathbf{y}) w(\mathbf{x}_{m}) w(\mathbf{y}) \\
            & + \frac{1}{2 M^{2}} \sum_{m = 1}^{M} \sum_{j = 1}^{M} k(\mathbf{x}_{m}, \mathbf{x}_{j}) w(\mathbf{x}_{m}) w(\mathbf{x}_{j}) + \frac{1}{2} k(\mathbf{y}, \mathbf{y})w(\mathbf{y})w(\mathbf{y}),
        \end{aligned}

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
    ens_w : array_like
        Weights assigned to the ensemble members. Array with one less dimension than fct (without the v_axis dimension).
        Default is equal weighting. Weights are normalised so that they sum to one across the ensemble members.
    backend : str, optional
        Deprecated, removed in 1.0. The array framework is inferred from the inputs;
        use ``scoringrules.set_config`` or ``scoringrules.config_context`` to control
        numba.

    Returns
    -------
    score : array_like
        The computed Vertically Re-scaled Gaussian Kernel Score.
    """
    resolve_backend_arg(backend)
    numba = use_numba("vrgksmv_ensemble", backend, obs, fct, aux=(ens_w,))
    xp = get_namespace(obs, fct)
    obs, fct = multivariate_array_check(obs, fct, m_axis, v_axis, xp=xp)
    obs_w, fct_w = mv_weighted_score_weights(obs, fct, w_func, xp=xp)
    numba = numba and gufunc_inputs_supported(obs_w, fct_w)  # e.g. masked w_func output
    if ens_w is not None:
        ens_w = multivariate_weight_check(ens_w, fct, m_axis, v_axis, xp=xp)

    if numba:
        if ens_w is None:
            return kernels.estimator_gufuncs_mv["vr"](obs, fct, obs_w, fct_w)
        return kernels.estimator_gufuncs_mv_w["vr"](obs, fct, obs_w, fct_w, ens_w)
    if ens_w is None:
        return kernels.vr_ensemble_mv(obs, fct, obs_w, fct_w, xp=xp)
    return kernels.vr_ensemble_mv_w(obs, fct, obs_w, fct_w, ens_w, xp=xp)


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
