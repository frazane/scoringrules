import typing as tp

from scoringrules.core import logarithmic

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike, Backend


def logs_beta(
    observation: "ArrayLike",
    a: "ArrayLike",
    b: "ArrayLike",
    /,
    lower: "ArrayLike" = 0.0,
    upper: "ArrayLike" = 1.0,
    *,
    backend: "Backend" = None,
) -> "ArrayLike":
    r"""Compute the logarithmic score (LS) for the beta distribution.

    This score is equivalent to the negative log likelihood of the beta distribution.

    Parameters
    ----------
    observation:
        The observed values.
    a:
        First shape parameter of the forecast beta distribution.
    b:
        Second shape parameter of the forecast beta distribution.
    lower:
        Lower bound of the forecast beta distribution.
    upper:
        Upper bound of the forecast beta distribution.
    backend:
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    score:
        The LS between Beta(a, b) and obs.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.logs_beta(0.3, 0.7, 1.1)
    """
    return logarithmic.beta(observation, a, b, lower, upper, backend=backend)


def logs_exponential2(
    observation: "ArrayLike",
    /,
    location: "ArrayLike" = 0.0,
    scale: "ArrayLike" = 1.0,
    *,
    backend: "Backend" = None,
) -> "ArrayLike":
    r"""Compute the logarithmic score (LS) for the exponential distribution with location and scale parameters.

    This score is equivalent to the negative log likelihood of the exponential distribution.

    Parameters
    ----------
    observation:
        The observed values.
    location:
        Location parameter of the forecast exponential distribution.
    scale:
        Scale parameter of the forecast exponential distribution.
    backend:
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    score:
        The LS between obs and Exp2(location, scale).

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.logs_exponential2(0.2, 0.0, 1.0)
    """
    return logarithmic.exponential2(observation, location, scale, backend=backend)


def logs_2pexponential(
    observation: "ArrayLike",
    scale1: "ArrayLike",
    scale2: "ArrayLike",
    location: "ArrayLike",
    /,
    *,
    backend: "Backend" = None,
) -> "ArrayLike":
    r"""Compute the logarithmic score (LS) for the two-piece exponential distribution.

    This score is equivalent to the negative log likelihood of the two-piece exponential distribution.

    Parameters
    ----------
    observation:
        The observed values.
    scale1:
        First scale parameter of the forecast two-piece exponential distribution.
    scale2:
        Second scale parameter of the forecast two-piece exponential distribution.
    location:
        Location parameter of the forecast two-piece exponential distribution.
    backend:
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    score:
        The LS between 2pExp(sigma1, sigma2, location) and obs.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.logs_2pexponential(0.8, 3.0, 1.4, 0.0)
    """
    return logarithmic.twopexponential(
        observation, scale1, scale2, location, backend=backend
    )


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


def logs_2pnormal(
    observation: "ArrayLike",
    scale1: "ArrayLike",
    scale2: "ArrayLike",
    location: "ArrayLike",
    /,
    *,
    backend: "Backend" = None,
) -> "ArrayLike":
    r"""Compute the logarithmic score (LS) for the two-piece normal distribution.

    This score is equivalent to the negative log likelihood of the two-piece normal distribution.

    Parameters
    ----------
    observations: ArrayLike
        The observed values.
    scale1: ArrayLike
        Scale parameter of the lower half of the forecast two-piece normal distribution.
    scale2: ArrayLike
        Scale parameter of the upper half of the forecast two-piece normal distribution.
    location: ArrayLike
        Location parameter of the forecast two-piece normal distribution.
    backend:
        The name of the backend used for computations. Defaults to 'numba' if available, else 'numpy'.

    Returns
    -------
    score:
        The LS between 2pNormal(scale1, scale2, location) and obs.
    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.logs_2pnormal(0.0, 0.4, 2.0, 0.1)
    """
    return logarithmic.twopnormal(
        observation, scale1, scale2, location, backend=backend
    )
