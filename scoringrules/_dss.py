import typing as tp

from scoringrules.backend import backends
from scoringrules.core import dss
from scoringrules.core.utils import multivariate_array_check

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, Backend


def dss_ensemble(
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

    ## TO-DO ##

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
    backend : str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    ds_score: Array
        The computed Dawid-Sebastiani-Score.
    """
    backend = backend if backend is not None else backends._active
    obs, fct = multivariate_array_check(obs, fct, m_axis, v_axis, backend=backend)

    if backend == "numba":
        return dss._dss_gufunc(obs, fct, bias)

    return dss.dss(obs, fct, bias, backend=backend)
