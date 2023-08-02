import numpy as np
import typing as tp

from scoringrules.backend import backends as srb
from scoringrules.backend.arrayapi import Array

ArrayLike = tp.TypeVar("ArrayLike", Array, float)


def twcrps_ensemble(
    forecasts: Array,
    observations: ArrayLike,
    /,
    v_func: tp.Callable=lambda x, *args: x,
    v_funcargs: tuple=(),
    *,
    backend: str = "numba",
) -> Array:
    r"""Estimate the Threshold-Weighted Continuous Ranked Probability Score (twCRPS) for a finite ensemble.

    Parameters
    ----------
    forecasts: ArrayLike
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the last axis.
    observations: ArrayLike
        The observed values.
    v_func: tp.Callable
        Chaining function used to emphasise particular outcomes.
    v_funcargs: tuple
        Additional arguments to the chaining function.
    axis: int
        The axis corresponding to the ensemble. Default is the last axis.
    sorted_ensemble: bool
        Boolean indicating whether the ensemble members are already in ascending order.
        Default is False.
    estimator: str
        Indicates the CRPS estimator to be used.
    backend: str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    twcrps: ArrayLike
        The twCRPS between the forecast ensemble and obs for the chosen weight function.

    Examples
    --------
    >>> from scoringrules import crps
    >>> twcrps.ensemble(pred, obs)
    """
    return srb[backend].twcrps_ensemble(
        forecasts,
        observations,
        v_func,
        v_funcargs,
    )
    

def owcrps_ensemble(
    forecasts: Array,
    observations: ArrayLike,
    /,
    w_func: tp.Callable=lambda x, *args: np.ones_like(x),
    w_funcargs: tuple=(),
    *,
    backend: str = "numba",
) -> Array:
    r"""Estimate the Outcome-Weighted Continuous Ranked Probability Score (owCRPS) for a finite ensemble.

    Parameters
    ----------
    forecasts: ArrayLike
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the last axis.
    observations: ArrayLike
        The observed values.
    w_func: tp.Callable
        Weight function used to emphasise particular outcomes.
    w_funcargs: tuple
        Additional arguments to the weight function.
    axis: int
        The axis corresponding to the ensemble. Default is the last axis.
    sorted_ensemble: bool
        Boolean indicating whether the ensemble members are already in ascending order.
        Default is False.
    estimator: str
        Indicates the CRPS estimator to be used.
    backend: str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    owcrps: ArrayLike
        The owCRPS between the forecast ensemble and obs for the chosen weight function.

    Examples
    --------
    >>> from scoringrules import crps
    >>> owcrps.ensemble(pred, obs)
    """
    return srb[backend].owcrps_ensemble(
        forecasts,
        observations,
        w_func,
        w_funcargs,
    )
    
    
__all__ = [
    "twcrps_ensemble",
    "owcrps_ensemble",
]