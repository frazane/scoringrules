import typing as tp

from scoringrules.backend import backends as srb
from scoringrules.backend.arrayapi import Array

ArrayLike = tp.TypeVar("ArrayLike", Array, float)


def logscore_normal(
    mu: ArrayLike,
    sigma: ArrayLike,
    observation: ArrayLike,
    /,
    *,
    negative: bool = True,
    backend: tp.Literal["numba", "numpy", "jax"] = "numba",
) -> Array:
    r"""Compute the logarithmic score (LS) for the normal distribution.

    Parameters
    ----------
    mu: ArrayLike
        Mean of the forecast normal distribution.
    sigma: ArrayLike
        Standard deviation of the forecast normal distribution.
    observation: ArrayLike
        The observed values.

    Returns
    -------
    ls: array_like
        The LS between Normal(mu, sigma) and obs.

    Examples
    --------
    >>> from scoringrules import crps
    >>> crps.normal(0.1, 0.4, 0.0)
    """
    return srb[backend].logscore_normal(mu, sigma, observation, negative=negative)
