import typing as tp

from scoringrules.core import dss
from scoringrules.core.utils import multivariate_array_check

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, Backend


def dawid_sebastiani_score(
    observations: "Array",
    forecasts: "Array",
    /,
    m_axis: int = -2,
    v_axis: int = -1,
    *,
    backend: "Backend" = None,
) -> "Array":
    r"""Compute the Dawid-Sebastiani-Score for a finite multivariate ensemble.

    ## TO-DO ##

    Parameters
    ----------
    forecasts: Array
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the second last axis and the variables dimension by the last axis.
    observations: Array
        The observed values, where the variables dimension is by default the last axis.
    m_axis: int
        The axis corresponding to the ensemble dimension. Defaults to -2.
    v_axis: int
        The axis corresponding to the variables dimension. Defaults to -1.
    backend: str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    ds_score: Array
        The computed Dawid-Sebastiani-Score.
    """
    observations, forecasts = multivariate_array_check(
        observations, forecasts, m_axis, v_axis, backend=backend
    )

    if backend == "numba":
        return dss._dss_gufunc(observations, forecasts)

    return dss._dss(observations, forecasts, backend=backend)
