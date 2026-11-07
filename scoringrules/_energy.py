import typing as tp

from scoringrules._dispatch import resolve_backend_arg, use_numba
from scoringrules.backend import get_namespace
from scoringrules.core import energy
from scoringrules.core.utils import (
    apply_nan_policy_ens_mv,
    estimator_check,
    multivariate_array_check,
    multivariate_weight_check,
    mv_weighted_score_chain,
    mv_weighted_score_weights,
)

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike, Backend, NanPolicy


def es_ensemble(
    obs: "Array",
    fct: "Array",
    m_axis: int = -2,
    v_axis: int = -1,
    *,
    ens_w: "Array" = None,
    estimator: str = "nrg",
    nan_policy: "NanPolicy" = "propagate",
    backend: "Backend" = None,
) -> "Array":
    r"""Compute the Energy Score for a finite multivariate ensemble.

    The Energy Score is a multivariate scoring rule expressed as

    .. math::
        \text{ES}(F_{ens}, \mathbf{y})= \frac{1}{M} \sum_{m=1}^{M} \| \mathbf{x}_{m} -
        \mathbf{y} \| - \frac{1}{2 M^{2}} \sum_{m=1}^{M} \sum_{j=1}^{M} \| \mathbf{x}_{m} - \mathbf{x}_{j} \|,

    where :math:`||\cdot||` is the euclidean norm over the input dimensions (the variables).

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
        The energy score estimator to be used.
    nan_policy : {'propagate', 'omit', 'raise'}, default 'propagate'
        How to handle NaN ensemble members (a member is invalid if any of its
        variables, or its weight, is NaN). 'propagate' lets NaN flow to a NaN
        score; 'raise' errors on any NaN; 'omit' drops invalid members by
        zero-weighting them. 'omit' is not implemented for the 'akr' and
        'akr_circperm' estimators. NaN in `obs` always propagates.
    backend : str, optional
        Computational backend. ``None`` (default) infers the array framework from
        the inputs; pass ``"numba"`` for the numba fast path.

    Returns
    -------
    es_ensemble : array_like
        The computed Energy Score.

    See Also
    --------
    twes_ensemble, owes_ensemble, vres_ensemble
        Weighted variants of the Energy Score.
    crps_ensemble
        The univariate equivalent of the Energy Score.

    Notes
    -----
    :ref:`theory.multivariate`
        Some theoretical background on scoring rules for multivariate forecasts.
    """
    resolve_backend_arg(backend)
    numba = use_numba(backend, obs, fct)
    xp = get_namespace(obs, fct)
    obs, fct = multivariate_array_check(obs, fct, m_axis, v_axis, xp=xp)
    # apply the nan policy before normalising the weights, so NaN in a user ens_w
    # stays detectable. For 'omit', invalid members become zero-weighted and the
    # score goes through the weighted path. ens_w comes back ensemble-last.
    obs, fct, ens_w = apply_nan_policy_ens_mv(
        obs,
        fct,
        nan_policy,
        ens_w,
        estimator=estimator,
        m_axis=m_axis,
        v_axis=v_axis,
        xp=xp,
    )
    if ens_w is not None:
        # already ensemble-last: (-2, -1) only checks and normalises
        ens_w = multivariate_weight_check(ens_w, fct, -2, -1, xp=xp)

    if numba:
        if ens_w is None:
            estimator_check(estimator, energy.estimator_gufuncs)
            return energy.estimator_gufuncs[estimator](obs, fct)
        estimator_check(estimator, energy.estimator_gufuncs_w)
        return energy.estimator_gufuncs_w[estimator](obs, fct, ens_w)
    if ens_w is None:
        return energy.es(obs, fct, estimator=estimator, xp=xp)
    return energy.es_w(obs, fct, ens_w, estimator=estimator, xp=xp)


def twes_ensemble(
    obs: "Array",
    fct: "Array",
    v_func: tp.Callable[["ArrayLike"], "ArrayLike"],
    m_axis: int = -2,
    v_axis: int = -1,
    *,
    ens_w: "Array" = None,
    estimator: str = "nrg",
    nan_policy: "NanPolicy" = "propagate",
    backend: "Backend" = None,
) -> "Array":
    r"""Compute the Threshold-Weighted Energy Score (twES) for a finite multivariate ensemble.

    Computation is performed using the ensemble representation of the twES in
    [Allen et al. (2022)](https://arxiv.org/abs/2202.12732):

    .. math::
        \mathrm{twES}(F_{ens}, \mathbf{y}) = \frac{1}{M} \sum_{m = 1}^{M} \| v(\mathbf{x}_{m}) - v(\mathbf{y}) \|
        - \frac{1}{2 M^{2}} \sum_{m = 1}^{M} \sum_{j = 1}^{M} \| v(\mathbf{x}_{m}) - v(\mathbf{x}_{j}) \|,

    where :math:`F_{ens}` is the ensemble forecast :math:`\mathbf{x}_{1}, \dots, \mathbf{x}_{M}` with
    :math:`M` members, :math:`\| \cdotp \|` is the Euclidean distance, and :math:`v` is the chaining function
    used to target particular outcomes.


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
    v_axis : int or tuple of int
        The axis corresponding to the variables dimension. Defaults to -1.
    ens_w : array_like
        Weights assigned to the ensemble members. Array with one less dimension than fct (without the v_axis dimension).
        Default is equal weighting. Weights are normalised so that they sum to one across the ensemble members.
    estimator : str
        The energy score estimator to be used.
    nan_policy : {'propagate', 'omit', 'raise'}, default 'propagate'
        How to handle NaN ensemble members (a member is invalid if any of its
        variables, or its weight, is NaN). 'propagate' lets NaN flow to a NaN
        score; 'raise' errors on any NaN; 'omit' drops invalid members by
        zero-weighting them. 'omit' is not implemented for the 'akr' and
        'akr_circperm' estimators. NaN in `obs` always propagates.
    backend : str, optional
        Computational backend. ``None`` (default) infers the array framework from
        the inputs; pass ``"numba"`` for the numba fast path.

    Returns
    -------
    twes_ensemble : array_like
        The computed Threshold-Weighted Energy Score.
    """
    resolve_backend_arg(backend)
    # warned above, at the user's call: forward only the numba choice
    backend = "numba" if backend == "numba" else None
    obs, fct = mv_weighted_score_chain(obs, fct, v_func)
    return es_ensemble(
        obs,
        fct,
        m_axis=m_axis,
        v_axis=v_axis,
        ens_w=ens_w,
        estimator=estimator,
        nan_policy=nan_policy,
        backend=backend,
    )


def owes_ensemble(
    obs: "Array",
    fct: "Array",
    w_func: tp.Callable[["ArrayLike"], "ArrayLike"],
    m_axis: int = -2,
    v_axis: int = -1,
    *,
    ens_w: "Array" = None,
    nan_policy: "NanPolicy" = "propagate",
    backend: "Backend" = None,
) -> "Array":
    r"""Compute the Outcome-Weighted Energy Score (owES) for a finite multivariate ensemble.

    Computation is performed using the ensemble representation of the owES in
    [Allen et al. (2022)](https://arxiv.org/abs/2202.12732):

    .. math::
        \mathrm{owES}(F_{ens}, \mathbf{y}) = \frac{1}{M \bar{w}} \sum_{m = 1}^{M} \| \mathbf{x}_{m}
        - \mathbf{y} \| w(\mathbf{x}_{m}) w(\mathbf{y})
        - \frac{1}{2 M^{2} \bar{w}^{2}} \sum_{m = 1}^{M} \sum_{j = 1}^{M} \| \mathbf{x}_{m}
        - \mathbf{x}_{j} \| w(\mathbf{x}_{m}) w(\mathbf{x}_{j}) w(\mathbf{y}),


    where :math:`F_{ens}` is the ensemble forecast :math:`\mathbf{x}_{1}, \dots, \mathbf{x}_{M}` with
    :math:`M` members, :math:`\| \cdotp \|` is the Euclidean distance, :math:`w` is the chosen weight function,
    and :math:`\bar{w} = \sum_{m=1}^{M}w(\mathbf{x}_{m})/M`.


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
    nan_policy : {'propagate', 'omit', 'raise'}, default 'propagate'
        How to handle NaN ensemble members (a member is invalid if any of its
        variables, or its weight, is NaN). 'propagate' lets NaN flow to a NaN
        score; 'raise' errors on any NaN; 'omit' drops invalid members by
        zero-weighting them. NaN in `obs` always propagates.
    backend : str, optional
        Computational backend. ``None`` (default) infers the array framework from
        the inputs; pass ``"numba"`` for the numba fast path.

    Returns
    -------
    owes_ensemble : array_like
        The computed Outcome-Weighted Energy Score.
    """
    resolve_backend_arg(backend)
    numba = use_numba(backend, obs, fct)
    xp = get_namespace(obs, fct)
    obs, fct = multivariate_array_check(obs, fct, m_axis, v_axis, xp=xp)
    # apply the nan policy before deriving the outcome weights, so the (now zeroed)
    # NaN members do not leak NaN into obs_w/fct_w; they carry zero ens_w anyway.
    obs, fct, ens_w = apply_nan_policy_ens_mv(
        obs, fct, nan_policy, ens_w, m_axis=m_axis, v_axis=v_axis, xp=xp
    )
    obs_w, fct_w = mv_weighted_score_weights(obs, fct, w_func, xp=xp)
    if ens_w is not None:
        ens_w = multivariate_weight_check(ens_w, fct, -2, -1, xp=xp)

    if numba:
        if ens_w is None:
            return energy.estimator_gufuncs["ownrg"](obs, fct, obs_w, fct_w)
        return energy.estimator_gufuncs_w["ownrg"](obs, fct, obs_w, fct_w, ens_w)
    if ens_w is None:
        return energy.owes(obs, fct, obs_w, fct_w, xp=xp)
    return energy.owes_w(obs, fct, obs_w, fct_w, ens_w, xp=xp)


def vres_ensemble(
    obs: "Array",
    fct: "Array",
    w_func: tp.Callable[["ArrayLike"], "ArrayLike"],
    *,
    ens_w: "Array" = None,
    nan_policy: "NanPolicy" = "propagate",
    m_axis: int = -2,
    v_axis: int = -1,
    backend: "Backend" = None,
) -> "Array":
    r"""Compute the Vertically Re-scaled Energy Score (vrES) for a finite multivariate ensemble.

    Computation is performed using the ensemble representation of the vrES in
    [Allen et al. (2022)](https://arxiv.org/abs/2202.12732):

    .. math::
        \begin{split}
            \mathrm{vrES}(F_{ens}, \mathbf{y}) = & \frac{1}{M} \sum_{m = 1}^{M} \| \mathbf{x}_{m}
                - \mathbf{y} \| w(\mathbf{x}_{m}) w(\mathbf{y}) - \frac{1}{2 M^{2}} \sum_{m = 1}^{M} \sum_{j = 1}^{M} \| \mathbf{x}_{m}
                - \mathbf{x}_{j} \| w(\mathbf{x}_{m}) w(\mathbf{x_{j}}) \\
                & + \left( \frac{1}{M} \sum_{m = 1}^{M} \| \mathbf{x}_{m} \| w(\mathbf{x}_{m})
                - \| \mathbf{y} \| w(\mathbf{y}) \right) \left( \frac{1}{M} \sum_{m = 1}^{M} w(\mathbf{x}_{m}) - w(\mathbf{y}) \right),
        \end{split}

    where :math:`F_{ens}` is the ensemble forecast :math:`\mathbf{x}_{1}, \dots, \mathbf{x}_{M}` with
    :math:`M` members, and :math:`w` is the weight function used to target particular outcomes.


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
    v_axis : int or tuple of int
        The axis corresponding to the variables dimension. Defaults to -1.
    ens_w : array_like
        Weights assigned to the ensemble members. Array with one less dimension than fct (without the v_axis dimension).
        Default is equal weighting. Weights are normalised so that they sum to one across the ensemble members.
    nan_policy : {'propagate', 'omit', 'raise'}, default 'propagate'
        How to handle NaN ensemble members (a member is invalid if any of its
        variables, or its weight, is NaN). 'propagate' lets NaN flow to a NaN
        score; 'raise' errors on any NaN; 'omit' drops invalid members by
        zero-weighting them. NaN in `obs` always propagates.
    backend : str, optional
        Computational backend. ``None`` (default) infers the array framework from
        the inputs; pass ``"numba"`` for the numba fast path.

    Returns
    -------
    vres_ensemble : array_like
        The computed Vertically Re-scaled Energy Score.
    """
    resolve_backend_arg(backend)
    numba = use_numba(backend, obs, fct)
    xp = get_namespace(obs, fct)
    obs, fct = multivariate_array_check(obs, fct, m_axis, v_axis, xp=xp)
    # apply the nan policy before deriving the outcome weights, so the (now zeroed)
    # NaN members do not leak NaN into obs_w/fct_w; they carry zero ens_w anyway.
    obs, fct, ens_w = apply_nan_policy_ens_mv(
        obs, fct, nan_policy, ens_w, m_axis=m_axis, v_axis=v_axis, xp=xp
    )
    obs_w, fct_w = mv_weighted_score_weights(obs, fct, w_func, xp=xp)
    if ens_w is not None:
        ens_w = multivariate_weight_check(ens_w, fct, -2, -1, xp=xp)

    if numba:
        if ens_w is None:
            return energy.estimator_gufuncs["vrnrg"](obs, fct, obs_w, fct_w)
        return energy.estimator_gufuncs_w["vrnrg"](obs, fct, obs_w, fct_w, ens_w)
    if ens_w is None:
        return energy.vres(obs, fct, obs_w, fct_w, xp=xp)
    return energy.vres_w(obs, fct, obs_w, fct_w, ens_w, xp=xp)
