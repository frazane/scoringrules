import typing as tp

from scoringrules.backend import backends
from scoringrules.core import error_spread

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike, Backend


def error_spread_score(
    forecasts: "Array",
    observations: "ArrayLike",
    /,
    axis: int = -1,
    *,
    backend: "Backend" = None,
) -> "Array":
    r"""Estimate the Continuous Ranked Probability Score (CRPS) for a finite ensemble.

    Parameters
    ----------
    forecasts: ArrayLike
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the last axis.
    observations: ArrayLike
        The observed values.
    axis: int
        The axis corresponding to the ensemble. Default is the last axis.
    backend: str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    crps: ArrayLike
        The CRPS between the forecast ensemble and obs.

    Examples
    --------
    >>> from scoringrules import crps
    >>> crps.ensemble(pred, obs)
    """
    B = backends.active if backend is None else backends[backend]
    forecasts, observations = map(B.asarray, (forecasts, observations))

    if axis != -1:
        forecasts = B.moveaxis(forecasts, axis, -1)

    return error_spread.error_spread_score(forecasts, observations, backend=backend)
