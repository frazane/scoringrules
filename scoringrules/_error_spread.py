import typing as tp

from scoringrules.backend import backends
from scoringrules.core import error_spread

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike, Backend


def error_spread_score(
    observations: "ArrayLike",
    forecasts: "Array",
    /,
    m_axis: int = -1,
    *,
    backend: "Backend" = None,
) -> "Array":
    r"""Compute the error-spread score [(Christensen et al., 2015)](https://doi.org/10.1002/qj.2375) for a finite ensemble.

    Parameters
    ----------
    observations: ArrayLike
        The observed values.
    forecasts: Array
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the last axis.
    m_axis: int
        The axis corresponding to the ensemble. Default is the last axis.
    backend: str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    - Array
        An array of error spread scores for each ensemble forecast, which should be averaged to get meaningful values.
    """
    B = backends.active if backend is None else backends[backend]
    observations, forecasts = map(B.asarray, (observations, forecasts))

    if m_axis != -1:
        forecasts = B.moveaxis(forecasts, m_axis, -1)

    if B.name == "numba":
        return error_spread._ess_gufunc(observations, forecasts)

    return error_spread.ess(observations, forecasts, backend=backend)
