import numpy as np
import typing as tp

from scoringrules.backend import backends as srb
from scoringrules.backend.arrayapi import Array

ArrayLike = tp.TypeVar("ArrayLike", Array, float)


def twcrps_ensemble(
    forecasts: Array,
    observations: ArrayLike,
    /,
    v_func: tp.Callable = lambda x, *args: x,
    v_funcargs: tuple = (),
    *,
    axis: int = -1,
    backend: str = "numba",
) -> Array:
    r"""Estimate the Threshold-Weighted Continuous Ranked Probability Score (twCRPS) for a finite ensemble.

    Computation is performed using the ensemble representation of the twCRPS in
    [Allen et al. (2022)](https://arxiv.org/abs/2202.12732):

    $$ \mathrm{twCRPS}(F_{ens}, y) = \frac{1}{M} \sum_{m = 1}^{M} |v(x_{m}) - v(y)| - \frac{1}{2 M^{2}} \sum_{m = 1}^{M} \sum_{j = 1}^{M} |v(x_{m}) - v(x_{j})|,$$

    where $F_{ens}(x) = \sum_{m=1}^{M} \one \{ x_{m} \leq x \}/M$ is the empirical 
    distribution function associated with an ensemble forecast $x_{1}, \dots, x_{M}$ with
    $M$ members, and $v$ is the chaining function used to target particular outcomes.

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
        axis=axis,
    )
    

def owcrps_ensemble(
    forecasts: Array,
    observations: ArrayLike,
    /,
    w_func: tp.Callable = lambda x, *args: np.ones_like(x),
    w_funcargs: tuple = (),
    *,
    axis: int = -1,
    backend: str = "numba",
) -> Array:
    r"""Estimate the Outcome-Weighted Continuous Ranked Probability Score (owCRPS) for a finite ensemble.
    
    Computation is performed using the ensemble representation of the owCRPS in
    [Allen et al. (2022)](https://arxiv.org/abs/2202.12732):

    $$ \mathrm{owCRPS}(F_{ens}, y) = \frac{1}{M \bar{w}} \sum_{m = 1}^{M} |x_{m} - y|w(x_{m})w(y) - \frac{1}{2 M^{2} \bar{w}^{2}} \sum_{m = 1}^{M} \sum_{j = 1}^{M} |x_{m} - x_{j}|w(x_{m})w(x_{j})w(y),$$

    where $F_{ens}(x) = \sum_{m=1}^{M} \one \{ x_{m} \leq x \}/M$ is the empirical 
    distribution function associated with an ensemble forecast $x_{1}, \dots, x_{M}$ with
    $M$ members, $w$ is the chosen weight function, and $\bar{w} = \sum_{m=1}^{M}w(x_{m})/M$.

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
        axis=axis,
    )
    
    
__all__ = [
    "twcrps_ensemble",
    "owcrps_ensemble",
]