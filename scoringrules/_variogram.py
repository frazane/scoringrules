from typing import TypeVar

from scoringrules.backend import backends as srb
from scoringrules.backend.arrayapi import Array

ArrayLike = TypeVar("ArrayLike", Array, float)


def variogram_score(
    forecasts: Array,
    observations: Array,
    p: float = 1.0,
    m_axis: int = -1,
    v_axis: int = -2,
    backend="numba",
) -> Array:
    r"""Compute the Variogram Score for a finite multivariate ensemble.

    For a $D$-variate ensemble the Variogram Score
    [(Sheuerer and Hamill, 2015)](https://journals.ametsoc.org/view/journals/mwre/143/4/mwr-d-14-00269.1.xml#bib9)
    of order $p$ is expressed as

    $$VS(F, \mathbf{y}) = \sum_{i,j=1}^{D}(|y_i - y_j|^p - E_F|X_i - X_j|^p)^2 ,$$

    where $\mathbf{X}$ and $\mathbf{X'}$ are independently sampled ensembles from from $F$.


    Parameters
    ----------
    forecasts: Array
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the second last axis and the variables dimension by the last axis.
    observations: Array
        The observed values, where the variables dimension is by default the last axis.
    p: float
        The order of the Variogram Score. Typical values are 0.5, 1.0 or 2.0. Defaults to 1.0.
    m_axis: int
        The axis corresponding to the ensemble dimension. Defaults to -2.
    v_axis: int
        The axis corresponding to the variables dimension. Defaults to -1.
    backend: str
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    variogram_score: Array
        The computed Variogram Score.
    """
    return srb[backend].variogram_score(
        forecasts, observations, p=p, m_axis=m_axis, v_axis=v_axis
    )
