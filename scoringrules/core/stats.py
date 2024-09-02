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
    return B.maximum(1 - B.exp(-rate * x), B.asarray(0.0))


def _gamma_cdf(
    x: "ArrayLike", shape: "ArrayLike", rate: "ArrayLike", backend: "Backend" = None
) -> "Array":
    """Cumulative distribution function for the gamma distribution."""
    B = backends.active if backend is None else backends[backend]
    zero = B.asarray(0.0)
    return B.maximum(B.gammainc(shape, rate * B.maximum(x, zero)), zero)


def _pois_cdf(x: "ArrayLike", mean: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Cumulative distribution function for the Poisson distribution."""
    B = backends.active if backend is None else backends[backend]
    return B.max(B.ui_gamma(B.floor(x + 1), mean) / B.gamma(B.floor(x + 1)), 0)


def _pois_pdf(x: "ArrayLike", mean: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Probability mass function for the Poisson distribution."""
    B = backends.active if backend is None else backends[backend]
    return B.isinteger(x) * (mean**x * B.exp(-x) / B.factorial(x))


def _t_pdf(x: "ArrayLike", df: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Probability density function for the standard Student's t distribution."""
    B = backends.active if backend is None else backends[backend]
    return ((1 + x**2 / df) ** (-(df + 1) / 2)) / (B.sqrt(df) * B.beta(1 / 2, df / 2))


def _t_cdf(x: "ArrayLike", df: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Cumulative distribution function for the standard Student's t distribution."""
    B = backends.active if backend is None else backends[backend]
    return 1 / 2 + x * B.hypergeo(1 / 2, (df + 1) / 2, 3 / 2, -(x**2) / df) / (
        B.sqrt(df) * B.beta(1 / 2, df / 2)
    )


def _gev_cdf(x: "ArrayLike", shape: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Cumulative distribution function for the standard GEV distribution."""
    B = backends.active if backend is None else backends[backend]
    F_0 = B.exp(-B.exp(-x))
    F_p = B.ispositive(x + 1 / shape) * B.exp(-((1 + shape * x) ** (-1 / shape)))
    F_m = B.isnegative(x + 1 / shape) * B.exp(-((1 + shape * x) ** (-1 / shape))) + (
        1 - B.isnegative(x + 1 / shape)
    )
    F_xi = B.iszero(shape) * F_0 + B.ispositive(shape) * F_p + B.isnegative(shape) * F_m
    return F_xi


def _gpd_cdf(x: "ArrayLike", shape: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Cumulative distribution function for the standard GPD distribution."""
    B = backends.active if backend is None else backends[backend]

    # masks to handle the different cases
    shape_0 = shape == 0
    shape = B.where(shape_0, B.nan, shape)
    shape_p = shape > 0
    shape_n = ~(shape_0 | shape_p)
    x_pos = x >= 0
    x_gt_invxi = x > -1 / shape
    x = B.where(x_pos, x, B.nan)
    shape = B.where(shape_n & x_gt_invxi, B.nan, shape)

    cdf = 0.0
    cdf = B.where(shape_0 & x_pos, 1 - B.exp(-x), cdf)
    cdf = B.where(
        (shape_n & x_pos) | (shape_p & x_pos),
        1 - (1 + shape * x) ** (-1 / shape),
        cdf,
    )
    cdf = B.where(shape_n & x_gt_invxi, 1.0, cdf)
    return cdf


def _binom_pdf(
    k: "ArrayLike", n: "ArrayLike", prob: "ArrayLike", backend: "Backend" = None
) -> "Array":
    """Probability mass function for the binomial distribution."""
    B = backends.active if backend is None else backends[backend]
    return B.comb(n, k) * prob**k * (1 - prob) ** (n - k)


def _binom_cdf(
    k: "ArrayLike", n: "ArrayLike", prob: "ArrayLike", backend: "Backend" = None
) -> "Array":
    """Cumulative distribution function for the binomial distribution."""
    B = backends.active if backend is None else backends[backend]
    return B.where(k < 0, 0.0, B.betainc(n - B.minimum(k, n) + 1e-36, k + 1, 1 - prob))


def _hypergeo_pdf(
    x: "ArrayLike",
    m: "ArrayLike",
    n: "ArrayLike",
    prob: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Probability mass function for the hypergeometric distribution."""
    B = backends.active if backend is None else backends[backend]
    k = prob
    ind = (
        B.isinteger(x)
        * B.ispositive(x)
        * (1 - B.isnegative(x - B.maximum(0, k - n)))
        * (1 - B.isnegative(B.minimum(k, m) - x))
    )
    return ind * B.comb(m, x) * B.comb(n, k - x) / B.comb(m + n, k)


def _negbinom_cdf(
    x: "ArrayLike", n: "ArrayLike", prob: "ArrayLike", backend: "Backend" = None
) -> "Array":
    """Cumulative distribution function for the negative binomial distribution."""
    B = backends.active if backend is None else backends[backend]
    return (1 - B.isnegative(x)) * B.betainc(n, B.floor(x + 1), prob)
