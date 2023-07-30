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
    backend: tp.Literal["numba", "numpy", "jax"] = "numpy",
) -> Array:
    r"""Compute the logarithmic score (LS) for the normal distribution.

    This score is equivalent to the negative log likelihood.

    Parameters
    ----------
    mu: ArrayLike
        Mean of the forecast normal distribution.
    sigma: ArrayLike
        Standard deviation of the forecast normal distribution.
    observation: ArrayLike
        The observed values.
    backend: str, optional
        The backend used for computations.

    Returns
    -------
    ls: array_like
        The LS between Normal(mu, sigma) and obs.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.logscore_normal(0.1, 0.4, 0.0)
    >>> 0.033898
    """
    return srb[backend].logscore_normal(mu, sigma, observation)
