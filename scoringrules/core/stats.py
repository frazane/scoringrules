import typing as tp

from scoringrules.backend import backends

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike, Backend


def _norm_cdf(x: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Cumulative distribution function for the standard normal distribution."""
    B = backends.active if backend is None else backends[backend]
    return (1.0 + B.erf(x / B.sqrt(2.0))) / 2.0


def _norm_pdf(x: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Probability density function for the standard normal distribution."""
    B = backends.active if backend is None else backends[backend]
    return (1.0 / B.sqrt(2.0 * B.pi)) * B.exp(-(x**2) / 2)


def _logis_cdf(x: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Cumulative distribution function for the standard logistic distribution."""
    B = backends.active if backend is None else backends[backend]
    return 1 / (1 + B.exp(-x))


def _exp_cdf(x: "ArrayLike", rate: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Cumulative distribution function for the exponential distribution."""
    B = backends.active if backend is None else backends[backend]
    return B.maximum(1 - B.exp(- rate * x), 0)


def _gamma_cdf(x: "ArrayLike", shape: "ArrayLike", rate: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Cumulative distribution function for the gamma distribution."""
    B = backends.active if backend is None else backends[backend]
    return B.maximum(B.li_gamma(shape, rate * x) / B.gamma(shape), 0)


def _pois_cdf(x: "ArrayLike", mean: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Cumulative distribution function for the Poisson distribution."""
    B = backends.active if backend is None else backends[backend]
    return B.maximum(B.ui_gamma(B.floor(x + 1), mean) / B.gamma(B.floor(x + 1)), 0)


def _pois_pdf(x: "ArrayLike", mean: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Probability mass function for the Poisson distribution."""
    B = backends.active if backend is None else backends[backend]
    return B.isinteger(x) * (mean**x * B.exp(-x) / B.factorial(x))
