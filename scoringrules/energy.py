from typing import Any, TypeVar

import numpy as np

from scoringrules.backend import backends as srb

Array = TypeVar("Array", np.ndarray, Any)
ArrayLike = TypeVar("ArrayLike", np.ndarray, Any, float)


def energy_score(
    forecasts: Array,
    observations: Array,
    m_axis: int = -2,
    v_axis: int = -1,
    engine="numba",
) -> ArrayLike:
    """Compute the Energy Score for a finite multivariate ensemble.

    The Energy Score is a multivariate scoring rule expressed as

    $$ES(F, \\mathbf{y}) = E_{F} ||\\mathbf{X} - \\mathbf{y}||
    - \frac{1}{2}E_{F} ||\\mathbf{X} - \\mathbf{X'}||,$$

    where $\\mathbf{X}$ and $\\mathbf{X'}$ are independent samples from $F$
    and $||\\cdot|| is the euclidean norm over the input dimensions (the variables).


    Parameters
    ----------
    forecasts: ArrayLike of shape (..., D, M)
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the second last axis and the variables dimension by the last axis.
    observations: ArrayLike of shape (...,D)
        The observed values, where the variables dimension is by default the last axis.
    m_axis: int
        The axis corresponding to the ensemble dimension. Defaults to -2.
    v_axis: int or tuple(int)
        The axis corresponding to the variables dimension. Defaults to -1.

    Returns
    -------
    energy_score: ArrayLike of shape (...)
        The computed Energy Score.
    """
    return srb[engine].energy_score(
        forecasts, observations, m_axis=m_axis, v_axis=v_axis
    )
