import typing as tp

from scoringrules.backend import backends

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike, Backend


def _norm_pdf(x: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Probability density function for the standard normal distribution."""
    B = backends.active if backend is None else backends[backend]
    return (1.0 / B.sqrt(2.0 * B.pi)) * B.exp(-(x**2) / 2)


def _norm_cdf(x: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Cumulative distribution function for the standard normal distribution."""
    B = backends.active if backend is None else backends[backend]
    return (1.0 + B.erf(x / B.sqrt(2.0))) / 2.0


def _laplace_pdf(x: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Probability density function for the standard laplace distribution."""
    B = backends.active if backend is None else backends[backend]
    return B.exp(-B.abs(x)) / 2.0


def _logis_pdf(x: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Probability density function for the standard logistic distribution."""
    B = backends.active if backend is None else backends[backend]
    return B.exp(-x) / (1 + B.exp(-x)) ** 2


def _logis_pdf(x: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Probability density function for the standard logistic distribution."""
    B = backends.active if backend is None else backends[backend]
    return B.exp(-x) / (1 + B.exp(-x)) ** 2


def _logis_cdf(x: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Cumulative distribution function for the standard logistic distribution."""
    B = backends.active if backend is None else backends[backend]
    return 1 / (1 + B.exp(-x))


def _exp_pdf(x: "ArrayLike", rate: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Probability density function for the exponential distribution."""
    B = backends.active if backend is None else backends[backend]
    return B.where(x < 0.0, 0.0, rate * B.exp(-rate * x))


def _exp_cdf(x: "ArrayLike", rate: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Cumulative distribution function for the exponential distribution."""
    B = backends.active if backend is None else backends[backend]
    return B.maximum(1 - B.exp(-rate * x), B.asarray(0.0))


def _gamma_pdf(
    x: "ArrayLike",
    shape: "ArrayLike",
    rate: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Probability density function for the gamma distribution."""
    B = backends.active if backend is None else backends[backend]
    prob = (rate**shape) * (x ** (shape - 1)) * (B.exp(-rate * x)) / B.gamma(shape)
    return B.where(x <= 0.0, 0.0, prob)


def _gamma_cdf(
    x: "ArrayLike",
    shape: "ArrayLike",
    rate: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Cumulative distribution function for the gamma distribution."""
    B = backends.active if backend is None else backends[backend]
    zero = B.asarray(0.0)
    return B.maximum(B.gammainc(shape, rate * B.maximum(x, zero)), zero)


def _pois_pdf(x: "ArrayLike", mean: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Probability mass function for the Poisson distribution."""
    B = backends.active if backend is None else backends[backend]
    x_plus = B.abs(x)
    d = B.where(
        B.floor(x_plus) < x_plus,
        0.0,
        mean ** (x_plus) * B.exp(-mean) / B.factorial(x_plus),
    )
    return B.where(mean < 0.0, B.nan, B.where(x < 0.0, 0.0, d))


def _pois_cdf(x: "ArrayLike", mean: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Cumulative distribution function for the Poisson distribution."""
    B = backends.active if backend is None else backends[backend]
    x_plus = B.abs(x)
    p = B.gammauinc(B.floor(x_plus + 1), mean) / B.gamma(B.floor(x_plus + 1))
    return B.where(x < 0.0, 0.0, p)


def _t_pdf(x: "ArrayLike", df: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Probability density function for the standard Student's t distribution."""
    B = backends.active if backend is None else backends[backend]
    x_inf = B.abs(x) == float("inf")
    x = B.where(x_inf, B.nan, x)
    s = ((1 + x**2 / df) ** (-(df + 1) / 2)) / (B.sqrt(df) * B.beta(1 / 2, df / 2))
    return B.where(x_inf, 0.0, s)


def _t_cdf(x: "ArrayLike", df: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Cumulative distribution function for the standard Student's t distribution."""
    B = backends.active if backend is None else backends[backend]
    x_minf = x == float("-inf")
    x_inf = x == float("inf")
    x = B.where(x_inf, B.nan, x)
    s = 1 / 2 + x * B.hypergeometric(1 / 2, (df + 1) / 2, 3 / 2, -(x**2) / df) / (
        B.sqrt(df) * B.beta(1 / 2, df / 2)
    )
    s = B.where(x_minf, 0.0, s)
    s = B.where(x_inf, 1.0, s)
    return s


def _gev_pdf(s: "ArrayLike", xi: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Probability density function for the standard GEV distribution."""
    B = backends.active if backend is None else backends[backend]

    xi_0 = xi == 0.0
    supp = (xi != 0.0) & (s * xi > -1)
    supp = supp | xi_0
    xi = B.where(xi_0, B.nan, xi)
    s = B.where(supp, s, B.nan)
    power = -(xi + 1) / xi

    pdf0 = B.exp(-s) * B.exp(-B.exp(-s))
    pdf = (1 + s * xi) ** power * B.exp(-((1 + s * xi) ** (-1 / xi)))
    pdf = B.where(xi_0, pdf0, pdf)
    pdf = B.where(supp, pdf, 0.0)

    return pdf


def _gev_cdf(s: "ArrayLike", xi: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Cumulative distribution function for the standard GEV distribution."""
    B = backends.active if backend is None else backends[backend]
    zero_shape = xi == 0
    xi = B.where(zero_shape, B.nan, xi)
    general_case = ~zero_shape & (xi * s > -1)
    cdf = B.nan * s
    cdf = B.where(zero_shape, B.exp(-B.exp(-s)), cdf)  # Gumbel CDF
    cdf = B.where(
        general_case,
        B.exp(-((1 + xi * B.where(general_case, s, B.nan)) ** (-1 / xi))),
        cdf,
    )  # General CDF
    cdf = B.where((xi > 0) & (s <= -1 / xi), 0, cdf)  # Lower bound CDF
    cdf = B.where((xi < 0) & (s >= 1 / B.abs(xi)), 1, cdf)  # Upper bound CDF
    return cdf


def _gpd_pdf(x: "ArrayLike", shape: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Probability density function for the standard GPD distribution."""
    B = backends.active if backend is None else backends[backend]

    pos_supp = (shape >= 0.0) & (x >= 0.0)
    shape_0 = shape == 0
    shape = B.where(shape_0, B.nan, shape)
    neg_supp = (shape < 0.0) & (x >= 0.0) & (x <= -1 / shape)
    supp = pos_supp | neg_supp
    x = B.where(supp, x, B.nan)
    power = -(shape + 1) / shape

    pdf = B.exp(-x)
    pdf = B.where(shape_0, pdf, (1 + shape * x) ** power)
    pdf = B.where(supp, pdf, 0.0)

    return pdf


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


def _hypergeo_pdf(k, M, n, N, backend=None):
    """
    Calculate the PMF of the hypergeometric distribution.

    We follow scipy.stats.hypergeom.pmf.

    k: number of observed successes.
    M: total population size.
    n: number of success states in the population.
    N: sample size (number of draws).
    """
    B = backends.active if backend is None else backends[backend]
    ind = (k >= B.maximum(B.asarray(0), N - M + n)) & (k <= B.minimum(n, N))
    return ind * B.comb(n, k) * B.comb(M - n, N - k) / B.comb(M, N)


def _hypergeo_cdf(k, M, n, N, backend=None):
    """Cumulative distribution function for the hypergeometric distribution."""
    B = backends.active if backend is None else backends[backend]

    def _inner(m, M, n, N):
        return B.sum(_hypergeo_pdf(m, M, n, N, backend=backend), axis=0)

    if k.size == 1:
        m = B.arange(k + 1)
        M, n, N = B.broadcast_arrays(M, n, N)
        return _inner(m[:, None], M[None], n[None], N[None])
    else:
        k, M, n, N = B.broadcast_arrays(k, M, n, N)
        _iter = zip(k.ravel(), M.ravel(), n.ravel(), N.ravel(), strict=True)
        return B.asarray(
            [_inner(B.arange(0, _args[0] + 1), *_args[1:]) for _args in _iter]
        ).reshape(k.shape)


def _negbinom_pdf(
    x: "ArrayLike", n: "ArrayLike", prob: "ArrayLike", backend: "Backend" = None
) -> "Array":
    """Probability mass function for the negative binomial distribution."""
    B = backends.active if backend is None else backends[backend]
    x_plus = B.abs(x)
    d = B.where(
        B.floor(x_plus) < x_plus,
        0.0,
        B.comb(n + x - 1, x) * prob**n * (1 - prob) ** x,
    )
    return B.where(x < 0.0, 0.0, d)


def _negbinom_cdf(
    x: "ArrayLike", n: "ArrayLike", prob: "ArrayLike", backend: "Backend" = None
) -> "Array":
    """Cumulative distribution function for the negative binomial distribution."""
    B = backends.active if backend is None else backends[backend]
    return B.where(x < 0.0, 0.0, B.betainc(n, B.floor(x + 1), prob))
