import typing as tp

from scoringrules.core import logarithmic

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike, Backend


def logs_negbinom(
    observation: "ArrayLike",
    n: "ArrayLike",
    /,
    prob: "ArrayLike | None" = None,
    *,
    mu: "ArrayLike | None" = None,
    backend: "Backend" = None,
) -> "ArrayLike":
    r"""Compute the logarithmic score (LS) for the negative binomial distribution.

    This score is equivalent to the negative log likelihood of the negative binomial distribution.

    Parameters
    ----------
    observation: ArrayLike
        The observed values.
    n: ArrayLike
        Size parameter of the forecast negative binomial distribution.
    prob: ArrayLike
        Probability parameter of the forecast negative binomial distribution.
    mu: ArrayLike
        Mean of the forecast negative binomial distribution.

    Returns
    -------
    score:
        The LS between NegBinomial(n, prob) and obs.

    Examples
    --------
    >>> import scoringrules as sr
    >>> sr.logs_negbinom(2, 5, 0.5)

    Raises
    ------
    ValueError
        If both `prob` and `mu` are provided, or if neither is provided.
    """
    if (prob is None and mu is None) or (prob is not None and mu is not None):
        raise ValueError(
            "Either `prob` or `mu` must be provided, but not both or neither."
        )

    if prob is None:
        prob = n / (n + mu)

    return logarithmic.negbinom(observation, n, prob, backend=backend)


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
