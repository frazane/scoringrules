import typing as tp

from scoringrules.backend import backends
from scoringrules.core import energy
from scoringrules.core.utils import multivariate_array_check

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike, Backend


def es_ensemble(
    obs: "Array",
    fct: "Array",
    /,
    m_axis: int = -2,
    v_axis: int = -1,
    *,
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
    backend : str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

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
    backend = backend if backend is not None else backends._active
    obs, fct = multivariate_array_check(obs, fct, m_axis, v_axis, backend=backend)

    if backend == "numba":
        return energy._energy_score_gufunc(obs, fct)

    return energy.nrg(obs, fct, backend=backend)


def twes_ensemble(
    obs: "Array",
    fct: "Array",
    v_func: tp.Callable[["ArrayLike"], "ArrayLike"],
    /,
    m_axis: int = -2,
    v_axis: int = -1,
    *,
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
    backend : str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    twes_ensemble : array_like
        The computed Threshold-Weighted Energy Score.
    """
    obs, fct = map(v_func, (obs, fct))
    return es_ensemble(obs, fct, m_axis=m_axis, v_axis=v_axis, backend=backend)


def owes_ensemble(
    obs: "Array",
    fct: "Array",
    w_func: tp.Callable[["ArrayLike"], "ArrayLike"],
    /,
    m_axis: int = -2,
    v_axis: int = -1,
    *,
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
    backend : str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    owes_ensemble : array_like
        The computed Outcome-Weighted Energy Score.
    """
    B = backends.active if backend is None else backends[backend]

    obs, fct = multivariate_array_check(obs, fct, m_axis, v_axis, backend=backend)

    fct_weights = B.apply_along_axis(w_func, fct, -1)
    obs_weights = B.apply_along_axis(w_func, obs, -1)

    if B.name == "numba":
        return energy._owenergy_score_gufunc(obs, fct, obs_weights, fct_weights)

    return energy.ownrg(obs, fct, obs_weights, fct_weights, backend=backend)


def vres_ensemble(
    obs: "Array",
    fct: "Array",
    w_func: tp.Callable[["ArrayLike"], "ArrayLike"],
    /,
    *,
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
    backend : str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    vres_ensemble : array_like
        The computed Vertically Re-scaled Energy Score.
    """
    B = backends.active if backend is None else backends[backend]

    obs, fct = multivariate_array_check(obs, fct, m_axis, v_axis, backend=backend)

    fct_weights = B.apply_along_axis(w_func, fct, -1)
    obs_weights = B.apply_along_axis(w_func, obs, -1)

    if backend == "numba":
        return energy._vrenergy_score_gufunc(obs, fct, obs_weights, fct_weights)

    return energy.vrnrg(obs, fct, obs_weights, fct_weights, backend=backend)
