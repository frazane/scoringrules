"""Statistics helpers (pdfs, pmfs, cdfs).

Used by the CRPS scores (``core/crps/_closed.py`` and ``_crps.py``) and the
logarithmic scores (``core/logarithmic.py``). They take a keyword-only ``xp``
augmented array-API namespace (see ``scoringrules.backend.get_namespace``).
"""

import typing as tp

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike


def _norm_pdf(x: "ArrayLike", *, xp) -> "Array":
    """Probability density function for the standard normal distribution."""
    return (1.0 / xp.sqrt(2.0 * xp.pi)) * xp.exp(-(x**2) / 2)


def _norm_cdf(x: "ArrayLike", *, xp) -> "Array":
    """Cumulative distribution function for the standard normal distribution."""
    return (1.0 + xp.erf(x / xp.sqrt(2.0))) / 2.0


def _laplace_pdf(x: "ArrayLike", *, xp) -> "Array":
    """Probability density function for the standard laplace distribution."""
    return xp.exp(-xp.abs(x)) / 2.0


def _logis_pdf(x: "ArrayLike", *, xp) -> "Array":
    """Probability density function for the standard logistic distribution."""
    return xp.exp(-x) / (1 + xp.exp(-x)) ** 2


def _logis_cdf(x: "ArrayLike", *, xp) -> "Array":
    """Cumulative distribution function for the standard logistic distribution."""
    return 1 / (1 + xp.exp(-x))


def _exp_pdf(x: "ArrayLike", rate: "ArrayLike", *, xp) -> "Array":
    """Probability density function for the exponential distribution."""
    return xp.where(x < 0.0, 0.0, rate * xp.exp(-rate * x))


def _exp_cdf(x: "ArrayLike", rate: "ArrayLike", *, xp) -> "Array":
    """Cumulative distribution function for the exponential distribution."""
    cdf = 1 - xp.exp(-rate * x)
    return xp.maximum(cdf, xp.zeros_like(cdf))


def _gamma_pdf(
    x: "ArrayLike",
    shape: "ArrayLike",
    rate: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Probability density function for the gamma distribution."""
    prob = (rate**shape) * (x ** (shape - 1)) * (xp.exp(-rate * x)) / xp.gamma(shape)
    return xp.where(x <= 0.0, 0.0, prob)


def _gamma_cdf(
    x: "ArrayLike",
    shape: "ArrayLike",
    rate: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Cumulative distribution function for the gamma distribution."""
    x = xp.asarray(x)
    cdf = xp.gammainc(shape, rate * xp.maximum(x, xp.zeros_like(x)))
    return xp.maximum(cdf, xp.zeros_like(cdf))


def _pois_pdf(x: "ArrayLike", mean: "ArrayLike", *, xp) -> "Array":
    """Probability mass function for the Poisson distribution."""
    x_plus = xp.abs(x)
    d = xp.where(
        xp.floor(x_plus) < x_plus,
        0.0,
        mean ** (x_plus) * xp.exp(-mean) / xp.factorial(x_plus),
    )
    return xp.where(mean < 0.0, xp.nan, xp.where(x < 0.0, 0.0, d))


def _pois_cdf(x: "ArrayLike", mean: "ArrayLike", *, xp) -> "Array":
    """Cumulative distribution function for the Poisson distribution."""
    x_plus = xp.abs(x)
    p = xp.gammauinc(xp.floor(x_plus + 1), mean) / xp.gamma(xp.floor(x_plus + 1))
    return xp.where(x < 0.0, 0.0, p)


def _t_pdf(x: "ArrayLike", df: "ArrayLike", *, xp) -> "Array":
    """Probability density function for the standard Student's t distribution."""
    x_inf = xp.abs(x) == float("inf")
    x = xp.where(x_inf, xp.nan, x)
    half_df = df / 2
    # beta(1/2, df/2) with 1/2 as an array: torch's lgamma rejects Python floats.
    beta = xp.beta(xp.full_like(half_df, 0.5), half_df)
    s = ((1 + x**2 / df) ** (-(df + 1) / 2)) / (xp.sqrt(df) * beta)
    return xp.where(x_inf, 0.0, s)


def _t_cdf(x: "ArrayLike", df: "ArrayLike", *, xp) -> "Array":
    """Cumulative distribution function for the standard Student's t distribution."""
    t = df / (x**2 + df)
    ibeta = xp.betainc(df / 2.0, 0.5, t)
    s = xp.where(x >= 0, 1 - 0.5 * ibeta, 0.5 * ibeta)
    return s


def _gev_pdf(s: "ArrayLike", xi: "ArrayLike", *, xp) -> "Array":
    """Probability density function for the standard GEV distribution."""
    xi_0 = xi == 0.0
    supp = (xi != 0.0) & (s * xi > -1)
    supp = supp | xi_0
    xi = xp.where(xi_0, xp.nan, xi)
    s = xp.where(supp, s, xp.nan)
    power = -(xi + 1) / xi

    pdf0 = xp.exp(-s) * xp.exp(-xp.exp(-s))
    pdf = (1 + s * xi) ** power * xp.exp(-((1 + s * xi) ** (-1 / xi)))
    pdf = xp.where(xi_0, pdf0, pdf)
    pdf = xp.where(supp, pdf, 0.0)

    return pdf


def _gev_cdf(s: "ArrayLike", xi: "ArrayLike", *, xp) -> "Array":
    """Cumulative distribution function for the standard GEV distribution."""
    zero_shape = xi == 0
    xi = xp.where(zero_shape, xp.nan, xi)
    general_case = ~zero_shape & (xi * s > -1)
    cdf = xp.nan * s
    cdf = xp.where(zero_shape, xp.exp(-xp.exp(-s)), cdf)  # Gumbel CDF
    cdf = xp.where(
        general_case,
        xp.exp(-((1 + xi * xp.where(general_case, s, xp.nan)) ** (-1 / xi))),
        cdf,
    )  # General CDF
    cdf = xp.where((xi > 0) & (s <= -1 / xi), 0, cdf)  # Lower bound CDF
    cdf = xp.where((xi < 0) & (s >= 1 / xp.abs(xi)), 1, cdf)  # Upper bound CDF
    return cdf


def _gpd_pdf(x: "ArrayLike", shape: "ArrayLike", *, xp) -> "Array":
    """Probability density function for the standard GPD distribution."""
    pos_supp = (shape >= 0.0) & (x >= 0.0)
    shape_0 = shape == 0
    shape = xp.where(shape_0, xp.nan, shape)
    neg_supp = (shape < 0.0) & (x >= 0.0) & (x <= -1 / shape)
    supp = pos_supp | neg_supp
    x = xp.where(supp, x, xp.nan)
    power = -(shape + 1) / shape

    pdf = xp.exp(-x)
    pdf = xp.where(shape_0, pdf, (1 + shape * x) ** power)
    pdf = xp.where(supp, pdf, 0.0)

    return pdf


def _gpd_cdf(x: "ArrayLike", shape: "ArrayLike", *, xp) -> "Array":
    """Cumulative distribution function for the standard GPD distribution."""
    # masks to handle the different cases
    shape_0 = shape == 0
    shape = xp.where(shape_0, xp.nan, shape)
    shape_p = shape > 0
    shape_n = ~(shape_0 | shape_p)
    x_pos = x >= 0
    x_gt_invxi = x > -1 / shape
    x = xp.where(x_pos, x, xp.nan)
    shape = xp.where(shape_n & x_gt_invxi, xp.nan, shape)

    cdf = 0.0
    cdf = xp.where(shape_0 & x_pos, 1 - xp.exp(-x), cdf)
    cdf = xp.where(
        (shape_n & x_pos) | (shape_p & x_pos),
        1 - (1 + shape * x) ** (-1 / shape),
        cdf,
    )
    cdf = xp.where(shape_n & x_gt_invxi, 1.0, cdf)
    return cdf


def _binom_pdf(k: "ArrayLike", n: "ArrayLike", prob: "ArrayLike", *, xp) -> "Array":
    """Probability mass function for the binomial distribution."""
    return xp.comb(n, k) * prob**k * (1 - prob) ** (n - k)


def _binom_cdf(k: "ArrayLike", n: "ArrayLike", prob: "ArrayLike", *, xp) -> "Array":
    """Cumulative distribution function for the binomial distribution."""
    return xp.where(
        k < 0, 0.0, xp.betainc(n - xp.minimum(k, n) + 1e-36, k + 1, 1 - prob)
    )


def _hypergeo_pdf(k, M, n, N, *, xp):
    """
    Calculate the PMF of the hypergeometric distribution.

    We follow scipy.stats.hypergeom.pmf.

    k: number of observed successes.
    M: total population size.
    n: number of success states in the population.
    N: sample size (number of draws).
    """
    lo = N - M + n
    ind = (k >= xp.maximum(xp.zeros_like(lo), lo)) & (k <= xp.minimum(n, N))
    return ind * xp.comb(n, k) * xp.comb(M - n, N - k) / xp.comb(M, N)


def _hypergeo_cdf(k, M, n, N, *, xp):
    """Cumulative distribution function for the hypergeometric distribution."""

    def _inner(m, M, n, N):
        return xp.sum(_hypergeo_pdf(m, M, n, N, xp=xp), axis=0)

    # if k.size == 1:
    #     m = xp.arange(k + 1)
    #     M, n, N = xp.broadcast_arrays(M, n, N)
    #     return _inner(m[:, None], M[None], n[None], N[None])
    # else:
    k, M, n, N = xp.broadcast_arrays(k, M, n, N)
    _iter = zip(k.ravel(), M.ravel(), n.ravel(), N.ravel(), strict=True)
    dev = xp.device(k)
    # stack (not asarray): asarray on a list of torch tensors goes through
    # ``.item()``, i.e. a host round-trip that lands the result on the CPU.
    return xp.reshape(
        xp.stack(
            [
                _inner(xp.arange(0, _args[0] + 1, device=dev), *_args[1:])
                for _args in _iter
            ]
        ),
        k.shape,
    )


def _negbinom_pdf(x: "ArrayLike", n: "ArrayLike", prob: "ArrayLike", *, xp) -> "Array":
    """Probability mass function for the negative binomial distribution."""
    x_plus = xp.abs(x)
    # comb(n + x - 1, x) as 1 / ((n + x) * B(n, x + 1)), which is what
    # scipy.special.comb computes for real arguments (``xp.comb`` rounds to an
    # integer, wrong for a real-valued size ``n``). Like scipy's comb it is 0
    # when k > N or N < 0, i.e. for n < 1, as the former ``B.comb`` version did.
    big_n = n + x_plus - 1
    coef = xp.where(
        (x_plus <= big_n) & (big_n >= 0),
        1 / ((n + x_plus) * xp.beta(n, x_plus + 1)),
        0.0,
    )
    d = xp.where(
        xp.floor(x_plus) < x_plus,
        0.0,
        coef * prob**n * (1 - prob) ** x_plus,
    )
    return xp.where(x < 0.0, 0.0, d)


def _negbinom_cdf(x: "ArrayLike", n: "ArrayLike", prob: "ArrayLike", *, xp) -> "Array":
    """Cumulative distribution function for the negative binomial distribution."""
    return xp.where(x < 0.0, 0.0, xp.betainc(n, xp.floor(x + 1), prob))
