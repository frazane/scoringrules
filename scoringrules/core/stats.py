import typing as tp

from scoringrules.backend import backends

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike


def _norm_cdf(x: "ArrayLike", backend: str | None = None) -> "Array":
    """Cumulative distribution function for the standard normal distribution."""
    B = backends.active if backend is None else backends[backend]
    return (1.0 + B.erf(x / B.sqrt(2.0))) / 2.0


def _norm_pdf(x: "ArrayLike", backend: str | None = None) -> "Array":
    """Probability density function for the standard normal distribution."""
    B = backends.active if backend is None else backends[backend]
    return (1.0 / B.sqrt(2.0 * B.pi)) * B.exp(-(x**2) / 2)


def _logis_cdf(x: "ArrayLike", backend: str | None = None) -> "Array":
    """Cumulative distribution function for the standard logistic distribution."""
    B = backends.active if backend is None else backends[backend]
    return 1 / (1 + B.exp(-x))
