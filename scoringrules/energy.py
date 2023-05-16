import numpy as np
from numba import guvectorize
from numpy.typing import ArrayLike


def energy_score(
    forecasts: ArrayLike, observations: ArrayLike, m_axis: int = -2, v_axis: int = -1
):
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
    if v_axis != -1:
        forecasts = np.moveaxis(forecasts, v_axis, -1)
    if m_axis != -2:
        forecasts = np.moveaxis(forecasts, m_axis, -2)

    return _energy_score_gufunc(forecasts, observations)


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:])",
    ],
    "(m,d),(d)->()",
)
def _energy_score_gufunc(forecasts, observations, out):
    """Compute the Energy Score for a finite ensemble."""
    M = forecasts.shape[0]

    e_1 = 0
    e_2 = 0
    for i in range(M):
        for j in range(M):
            e_1 += np.linalg.norm(forecasts[i] - observations)
            e_2 += np.linalg.norm(forecasts[i] - forecasts[j])

    out[0] = e_1 - 0.5 / (M * (M - 1)) * e_2
