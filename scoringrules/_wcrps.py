import numpy as np
import typing as tp

from scoringrules.backend import backends as srb
from scoringrules.backend.arrayapi import Array

ArrayLike = tp.TypeVar("ArrayLike", Array, float)


def twcrps_ensemble(
    forecasts: Array,
    observations: ArrayLike,
    lower: float = -float('inf'),
    upper: float = float('inf'),
    /,
    *,
    axis: int = -1,
    sorted_ensemble: bool = False,
    estimator: str = "pwm",
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
    lower: float
        The lower bound in the weight function. Default is -Inf.
    upper: float
        The upper bound in the weight function. Default is Inf.
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
    >>> twcrps.ensemble(pred, obs, lower = -1.0, upper = 2.5)
    """
    return srb[backend].twcrps_ensemble(
        forecasts,
        observations,
        lower,
        upper,
        axis=axis,
        sorted_ensemble=sorted_ensemble,
        estimator=estimator,
    )
    

def owcrps_ensemble(
    forecasts: Array,
    observations: ArrayLike,
    w_func: tp.Callable=lambda x, *args: np.ones_like(x),
    v_funcargs: tuple=(),
    /,
    *,
    axis: int = -1,
    sorted_ensemble: bool = False,
    estimator: str = "pwm",
    backend: str = "numba",
) -> Array:
    r"""Estimate the Outcome-Weighted Continuous Ranked Probability Score (twCRPS) for a finite ensemble.

    Parameters
    ----------
    forecasts: ArrayLike
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the last axis.
    observations: ArrayLike
        The observed values.
    lower: float
        The lower bound in the weight function. Default is -Inf.
    upper: float
        The upper bound in the weight function. Default is Inf.
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
    >>> owcrps.ensemble(pred, obs, lower = -1.0, upper = 2.5)
    """
    return srb[backend].owcrps_ensemble(
        forecasts,
        observations,
        w_func,
        v_funcargs,
        axis=axis,
        sorted_ensemble=sorted_ensemble,
        estimator=estimator,
    )
    
    
__all__ = [
    "twcrps_ensemble",
    "owcrps_ensemble",
]