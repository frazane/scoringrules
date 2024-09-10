import typing as tp

from scoringrules.core import logarithmic

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike, Backend


def logs_laplace(
    observation: "ArrayLike",
    location: "ArrayLike" = 0.0,
    scale: "ArrayLike" = 1.0,
    /,
    *,
    backend: "Backend" = None,
) -> "ArrayLike":
    r"""Compute the logarithmic score (LS) for the Laplace distribution.

    This score is equivalent to the negative log likelihood of the Laplace distribution

    Parameters
    ----------
    observation:
        Observed values.
    location:
        Location parameter of the forecast laplace distribution.
    scale:
        Scale parameter of the forecast laplace distribution.

    Returns
    -------
    score:
        The LS between obs and Laplace(location, scale).

    >>> sr.logs_laplace(0.3, 0.1, 0.2)
    """
    return logarithmic.laplace(observation, location, scale, backend=backend)


def logs_loglaplace(
    observation: "ArrayLike",
    locationlog: "ArrayLike",
    scalelog: "ArrayLike",
    *,
    backend: "Backend" = None,
) -> "ArrayLike":
    r"""Compute the logarithmic score (LS) for the log-Laplace distribution.

    This score is equivalent to the negative log likelihood of the log-Laplace distribution

    Parameters
    ----------
    observation:
        Observed values.
    locationlog:
        Location parameter of the forecast log-laplace distribution.
    scalelog:
        Scale parameter of the forecast log-laplace distribution.

    Returns
    -------
    score:
        The LS between obs and Loglaplace(locationlog, scalelog).

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.logs_loglaplace(3.0, 0.1, 0.9)
    """
    return logarithmic.loglaplace(observation, locationlog, scalelog, backend=backend)


def logs_normal(
    observation: "ArrayLike",
    mu: "ArrayLike",
    sigma: "ArrayLike",
    /,
    *,
    negative: bool = True,
    backend: "Backend" = None,
) -> "Array":
    r"""Compute the logarithmic score (LS) for the normal distribution.

    This score is equivalent to the negative log likelihood (if `negative = True`)

    Parameters
    ----------
    observation: ArrayLike
        The observed values.
    mu: ArrayLike
        Mean of the forecast normal distribution.
    sigma: ArrayLike
        Standard deviation of the forecast normal distribution.
    backend: str, optional
        The backend used for computations.

    Returns
    -------
    ls: array_like
        The LS between Normal(mu, sigma) and obs.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.logs_normal(0.1, 0.4, 0.0)
    >>> 0.033898
    """
    return logarithmic.normal(
        observation, mu, sigma, negative=negative, backend=backend
    )
