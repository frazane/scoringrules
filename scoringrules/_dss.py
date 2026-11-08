import typing as tp

from scoringrules._dispatch import resolve_backend_arg, use_numba
from scoringrules.backend import get_namespace
from scoringrules.core import dss
from scoringrules.core.utils import multivariate_array_check, univariate_array_check

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, Backend


def dssuv_ensemble(
    obs: "Array",
    fct: "Array",
    m_axis: int = -1,
    *,
    bias: bool = False,
    backend: "Backend" = None,
) -> "Array":
    r"""Compute the Dawid-Sebastiani-Score for a finite univariate ensemble.

    The Dawid-Sebastiani Score for an ensemble forecast is defined as

    .. math::
        \text{DSS}(F_{ens}, y) = \frac{(y - \bar{x})^2}{\sigma^2} + 2 \log \sigma

    where :math:`\bar{x}` and :math:`\sigma` are the mean and standard deviation of the ensemble members.

    Parameters
    ----------
    obs : array_like
        The observed values.
    fct : array_like, shape (..., m)
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the last axis.
    m_axis : int
        The axis corresponding to the ensemble. Default is the last axis.
    bias : bool
        Logical specifying whether the biased or unbiased estimator of the standard deviation
        should be used to calculate the score. Default is the unbiased estimator (`bias=False`).
    backend : str, optional
        Computational backend. ``None`` (default) infers the array framework from
        the inputs; pass ``"numba"`` for the numba fast path.

    Returns
    -------
    score: Array
        The computed Dawid-Sebastiani Score.
    """
    resolve_backend_arg(backend)
    numba = use_numba(backend, obs, fct)
    xp = get_namespace(obs, fct)
    obs, fct = univariate_array_check(obs, fct, m_axis, xp=xp)
    if numba:
        return dss._dss_uv_gufunc(obs, fct, bias)
    return dss.ds_score_uv(obs, fct, bias, xp=xp)


def dssmv_ensemble(
    obs: "Array",
    fct: "Array",
    m_axis: int = -2,
    v_axis: int = -1,
    *,
    bias: bool = False,
    backend: "Backend" = None,
) -> "Array":
    r"""Compute the Dawid-Sebastiani-Score for a finite multivariate ensemble.

    The Dawid-Sebastiani Score for an ensemble forecast is defined as

    .. math::
        \text{DSS}(F_{ens}, \mathbf{y})= (\mathbf{y} - \bar{mathbf{x}})^{\top} \Sigma^-1 (\mathbf{y} - \bar{mathbf{x}}) + \log \det(\Sigma)

    where :math:`\bar{mathbf{x}}` is the mean of the ensemble members (along each dimension),
    and :math:`\Sigma` is the sample covariance matrix estimated from the ensemble members.
    The covariance matrix is only invertible when the ensemble has more members than
    variables (``m > d``). The score is NaN where the covariance is not positive
    definite to the precision of the input: ``m <= d``, collinear members, or NaN or
    infinite members.

    Parameters
    ----------
    obs : array_like
        The observed values, where the variables dimension is by default the last axis.
    fct : array_like
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the second last axis and the variables dimension by the last axis.
    m_axis : int
        The axis corresponding to the ensemble dimension. Defaults to -2.
    v_axis : int or tuple of int
        The axis corresponding to the variables dimension. Defaults to -1.
    bias : bool
        Logical specifying whether the biased or unbiased estimator of the covariance matrix
        should be used to calculate the score. Default is the unbiased estimator (`bias=False`).
    backend : str, optional
        Computational backend. ``None`` (default) infers the array framework from
        the inputs; pass ``"numba"`` for the numba fast path.

    Returns
    -------
    score: Array
        The computed Dawid-Sebastiani Score.
    """
    resolve_backend_arg(backend)
    numba = use_numba(backend, obs, fct)
    xp = get_namespace(obs, fct)
    obs, fct = multivariate_array_check(obs, fct, m_axis, v_axis, xp=xp)
    if numba:
        return dss._dss_mv_gufunc(obs, fct, bias)
    return dss.ds_score_mv(obs, fct, bias, xp=xp)
