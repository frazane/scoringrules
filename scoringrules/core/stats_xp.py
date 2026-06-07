"""Transitional copy of ``core/stats.py`` for the array-API migration.

These are ``xp``-parameterised copies of the statistics helpers needed by the
CRPS closed-form scores (``core/crps/_closed.py``). They are a faithful
translation of the corresponding helpers in ``core/stats.py``: instead of
resolving a backend object ``B`` from the registry, they take a keyword-only
``xp`` augmented array-API namespace (see ``scoringrules.backend.get_namespace``).

``core/stats.py`` is intentionally left untouched because it is still shared by
the not-yet-migrated logarithmic-score family.
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
    return xp.maximum(1 - xp.exp(-rate * x), xp.asarray(0.0))


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
    zero = xp.asarray(0.0)
    return xp.maximum(xp.gammainc(shape, rate * xp.maximum(x, zero)), zero)


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
    s = ((1 + x**2 / df) ** (-(df + 1) / 2)) / (xp.sqrt(df) * xp.beta(1 / 2, df / 2))
    return xp.where(x_inf, 0.0, s)


def _t_cdf(x: "ArrayLike", df: "ArrayLike", *, xp) -> "Array":
    """Cumulative distribution function for the standard Student's t distribution."""
    t = df / (x**2 + df)
    ibeta = xp.betainc(df / 2.0, 0.5, t)
    s = xp.where(x >= 0, 1 - 0.5 * ibeta, 0.5 * ibeta)
    return s


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
    ind = (k >= xp.maximum(xp.asarray(0), N - M + n)) & (k <= xp.minimum(n, N))
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
    return xp.asarray(
        [_inner(xp.arange(0, _args[0] + 1), *_args[1:]) for _args in _iter]
    ).reshape(k.shape)


def _negbinom_cdf(x: "ArrayLike", n: "ArrayLike", prob: "ArrayLike", *, xp) -> "Array":
    """Cumulative distribution function for the negative binomial distribution."""
    return xp.where(x < 0.0, 0.0, xp.betainc(n, xp.floor(x + 1), prob))
