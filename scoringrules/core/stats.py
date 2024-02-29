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
    return B.maximum(B.gammalinc(shape, rate * x) / B.gamma(shape), 0)


def _pois_cdf(x: "ArrayLike", mean: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Cumulative distribution function for the Poisson distribution."""
    B = backends.active if backend is None else backends[backend]
    return B.maximum(B.gammauinc(B.floor(x + 1), mean) / B.gamma(B.floor(x + 1)), 0)


def _pois_pdf(x: "ArrayLike", mean: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Probability mass function for the Poisson distribution."""
    B = backends.active if backend is None else backends[backend]
    return B.isinteger(x) * (mean**x * B.exp(-x) / B.factorial(x))

def _t_pdf(x: "ArrayLike", df: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Probability density function for the standard Student's t distribution."""
    B = backends.active if backend is None else backends[backend]
    return ((1 + x**2 / df)**(- (df + 1) / 2)) / (B.sqrt(df) * B.beta(1 / 2, df / 2))

def _t_cdf(x: "ArrayLike", df: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Cumulative distribution function for the standard Student's t distribution."""
    B = backends.active if backend is None else backends[backend]
    return 1 / 2 + x * B.hypergeometric(1 / 2, (df + 1) / 2, 3 / 2, - (x**2) / df) / (B.sqrt(df) * B.beta(1 / 2, df / 2))

def _gev_cdf(x: "ArrayLike", shape: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Cumulative distribution function for the standard GEV distribution."""
    B = backends.active if backend is None else backends[backend]
    F_0 = B.exp(- B.exp(-x))
    F_p = B.ispositive(x + 1 / shape) * B.exp(-(1 + shape * x)**(-1 / shape))
    F_m = B.isnegative(x + 1 / shape) * B.exp(-(1 + shape * x)**(-1 / shape)) + \
        (1 - B.isnegative(x + 1 / shape))
    F_xi = B.iszero(shape) * F_0 + B.ispositive(shape) * F_p + B.isnegative(shape) * F_m
    return F_xi

def _gpd_cdf(x: "ArrayLike", shape: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Cumulative distribution function for the standard GPD distribution."""
    B = backends.active if backend is None else backends[backend]
    F_0 = B.maximum(1 - B.exp(-x), 0)
    F_p = B.maximum(1 - (1 + shape * x)**(- 1 / shape), 0)
    F_m = B.isnegative(x - 1 / B.abs(shape)) * (1 - (1 + shape * x)**(- 1 / shape)) + \
        (1 - B.isnegative(x - 1 / B.abs(shape)))
    F_m = B.maximum(F_m, 0)
    F_xi = B.iszero(shape) * F_0 + B.ispositive(shape) * F_p + B.isnegative(shape) * F_m
    return F_xi

def _binom_pdf(x: "ArrayLike", n: "ArrayLike", prob: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Probability mass function for the binomial distribution."""
    B = backends.active if backend is None else backends[backend]
    ind = B.isinteger(x) * (1 - B.isnegative(x)) * (1 - B.negative(n - x))
    return ind * B.comb(n, x) * prob**x * (1 - prob)**(n - x)

def _binom_cdf(x: "ArrayLike", n: "ArrayLike", prob: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Cumulative distribution function for the binomial distribution."""
    B = backends.active if backend is None else backends[backend]
    return (1 - B.isnegative(x)) * B.betainc(n - B.floor(x), B.floor(x) + 1, 1 - prob)

def _hypergeo_pdf(x: "ArrayLike", m: "ArrayLike", n: "ArrayLike", prob: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Probability mass function for the hypergeometric distribution."""
    B = backends.active if backend is None else backends[backend]
    ind = B.isinteger(x) * B.ispositive(x) * (1 - B.isnegative(x - B.maximum(0, k - n))) * (1 - B.isnegative(B.minimum(k, m) - x))
    return ind * B.comb(m, x) * B.comb(n, k - x) / B.comb(m + n, k)

def _negbinom_cdf(x: "ArrayLike", n: "ArrayLike", prob: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Cumulative distribution function for the negative binomial distribution."""
    B = backends.active if backend is None else backends[backend]
    return (1 - B.isnegative(x)) * B.betainc(n, B.floor(x + 1), prob)