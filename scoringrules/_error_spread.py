import typing as tp

from scoringrules._dispatch import resolve_backend_arg, use_numba
from scoringrules.backend import get_namespace
from scoringrules.core import error_spread
from scoringrules.core.utils_xp import univariate_array_check

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike, Backend


def error_spread_score(
    obs: "ArrayLike",
    fct: "Array",
    m_axis: int = -1,
    *,
    backend: "Backend" = None,
) -> "Array":
    r"""Compute the error-spread score [(Christensen et al., 2015)](https://doi.org/10.1002/qj.2375) for a finite ensemble.

    Parameters
    ----------
    obs: ArrayLike
        The observed values.
    fct: Array
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the last axis.
    m_axis: int
        The axis corresponding to the ensemble. Default is the last axis.
    backend: str, optional
        Computational backend. ``None`` (default) infers the array framework from
        the inputs; pass ``"numba"`` for the numba fast path.

    Returns
    -------
    - Array
        An array of error spread scores for each ensemble forecast, which should be averaged to get meaningful values.
    """
    resolve_backend_arg(backend)
    numba = use_numba(backend, obs, fct)
    xp = get_namespace(obs, fct)
    obs, fct = univariate_array_check(obs, fct, m_axis, xp=xp)
    if numba:
        return error_spread._ess_gufunc(obs, fct)
    return error_spread.ess(obs, fct, xp=xp)
