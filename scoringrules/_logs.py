import typing as tp

from scoringrules.core import logarithmic

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike


def logs_normal(
    mu: "ArrayLike",
    sigma: "ArrayLike",
    observation: "ArrayLike",
    /,
    *,
    negative: bool = True,
    backend: tp.Literal["numpy", "jax", "torch"] | None = None,
) -> "Array":
    r"""Compute the logarithmic score (LS) for the normal distribution.

    This score is equivalent to the negative log likelihood (if `negative = True`)

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
    >>> sr.logs_normal(0.1, 0.4, 0.0)
    >>> 0.033898
    """
    return logarithmic.normal(
        mu, sigma, observation, negative=negative, backend=backend
    )
