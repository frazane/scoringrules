import typing as tp

from scoringrules.backend import backends
from scoringrules.core import dss
from scoringrules.core.utils import multivariate_array_check

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, Backend


def dssuv_ensemble(
    obs: "Array",
    fct: "Array",
    /,
    m_axis: int = -1,
    *,
    bias: bool = False,
    backend: "Backend" = None,
) -> "Array":
    r"""Compute the Dawid-Sebastiani-Score for a finite univariate ensemble.

    The Dawid-Sebastiani Score for an ensemble forecast is defined as

    .. math::
        \text{DSS}(F_{ens}, y)= \frac{(y - \bar{x)^2}{\sigma^2} + 2 \log \sigma

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
    backend : str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    score: Array
        The computed Dawid-Sebastiani Score.
    """
    B = backends.active if backend is None else backends[backend]
    obs, fct = map(B.asarray, (obs, fct))

    if m_axis != -1:
        fct = B.moveaxis(fct, m_axis, -1)

    if backend == "numba":
        return dss._dss_uv_gufunc(obs, fct, bias)

    return dss.ds_score_uv(obs, fct, bias, backend=backend)


def dssmv_ensemble(
    obs: "Array",
    fct: "Array",
    /,
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
    backend : str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    score: Array
        The computed Dawid-Sebastiani Score.
    """
    obs, fct = multivariate_array_check(obs, fct, m_axis, v_axis, backend=backend)

    if backend == "numba":
        return dss._dss_mv_gufunc(obs, fct, bias)

    return dss.ds_score_mv(obs, fct, bias, backend=backend)
