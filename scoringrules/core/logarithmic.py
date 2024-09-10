import typing as tp

from scoringrules.backend import backends
from scoringrules.core.stats import (
    _binom_pdf,
    _exp_pdf,
    _gamma_pdf,
    _hypergeo_pdf,
    _logis_pdf,
    _logis_cdf,
    _norm_pdf,
    _norm_cdf,
    _pois_pdf,
    _t_pdf,
    _t_cdf,
)

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike, Backend


def binomial(
    obs: "ArrayLike",
    n: "ArrayLike",
    prob: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Compute the logarithmic score for the binomial distribution."""
    B = backends.active if backend is None else backends[backend]
    obs, n, prob = map(B.asarray, (obs, n, prob))
    zind = obs < 0.0
    obs = B.where(zind, B.nan, obs)
    prob = _binom_pdf(obs, n, prob, backend=backend)
    s = B.where(zind, float("inf"), -B.log(prob))
    return s


def exponential(
    obs: "ArrayLike", rate: "ArrayLike", backend: "Backend" = None
) -> "Array":
    """Compute the logarithmic score for the exponential distribution."""
    B = backends.active if backend is None else backends[backend]
    rate, obs = map(B.asarray, (rate, obs))
    zind = obs < 0.0
    obs = B.where(zind, B.nan, obs)
    prob = _exp_pdf(obs, rate, backend=backend)
    s = B.where(zind, float("inf"), -B.log(prob))
    return s


def gamma(
    obs: "ArrayLike", shape: "ArrayLike", rate: "ArrayLike", backend: "Backend" = None
) -> "Array":
    """Compute the logarithmic score for the gamma distribution."""
    B = backends.active if backend is None else backends[backend]
    obs, shape, rate = map(B.asarray, (obs, shape, rate))
    prob = _gamma_pdf(obs, shape, rate, backend=backend)
    return -B.log(prob)


def hypergeometric(
    obs: "ArrayLike",
    m: "ArrayLike",
    n: "ArrayLike",
    k: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Compute the logarithmic score for the hypergeometric distribution."""
    B = backends.active if backend is None else backends[backend]
    obs, m, n, k = map(B.asarray, (obs, m, n, k))
    M = m + n
    N = k
    prob = _hypergeo_pdf(obs, M, m, N, backend=backend)
    return -B.log(prob)


def logistic(
    obs: "ArrayLike",
    mu: "ArrayLike",
    sigma: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Compute the logarithmic score for the logistic distribution."""
    B = backends.active if backend is None else backends[backend]
    mu, sigma, obs = map(B.asarray, (mu, sigma, obs))
    ω = (obs - mu) / sigma
    prob = _logis_pdf(ω, backend=backend) / sigma
    return -B.log(prob)


def tlogistic(
    obs: "ArrayLike",
    location: "ArrayLike",
    scale: "ArrayLike",
    lower: "ArrayLike",
    upper: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Compute the logarithmic score for the truncated logistic distribution."""
    B = backends.active if backend is None else backends[backend]
    obs, mu, sigma, lower, upper = map(B.asarray, (obs, location, scale, lower, upper))
    ω = (obs - mu) / sigma
    u = (upper - mu) / sigma
    l = (lower - mu) / sigma
    F_u = _logis_cdf(u, backend=backend)
    F_l = _logis_cdf(l, backend=backend)
    denom = F_u - F_l

    ind_out = (ω < l) | (ω > u)
    prob = _logis_pdf(ω) / sigma
    s = B.where(ind_out, float("inf"), -B.log(prob / denom))
    return s


def loglogistic(
    obs: "ArrayLike",
    mulog: "ArrayLike",
    sigmalog: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Compute the logarithmic score for the log-logistic distribution."""
    B = backends.active if backend is None else backends[backend]
    mulog, sigmalog, obs = map(B.asarray, (mulog, sigmalog, obs))
    zind = obs <= 0.0
    obs = B.where(zind, B.nan, obs)
    ω = (B.log(obs) - mulog) / sigmalog
    prob = _logis_pdf(ω, backend=backend) / sigmalog
    s = B.where(zind, float("inf"), -B.log(prob / obs))
    return s


def lognormal(
    obs: "ArrayLike",
    mulog: "ArrayLike",
    sigmalog: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Compute the logarithmic score for the log-normal distribution."""
    B = backends.active if backend is None else backends[backend]
    mulog, sigmalog, obs = map(B.asarray, (mulog, sigmalog, obs))
    zind = obs <= 0.0
    obs = B.where(zind, B.nan, obs)
    ω = (B.log(obs) - mulog) / sigmalog
    prob = _norm_pdf(ω, backend=backend) / sigmalog
    s = B.where(zind, float("inf"), -B.log(prob / obs))
    return s


def normal(
    obs: "ArrayLike",
    mu: "ArrayLike",
    sigma: "ArrayLike",
    negative: bool = True,
    backend: "Backend" = None,
) -> "Array":
    """Compute the logarithmic score for the normal distribution."""
    B = backends.active if backend is None else backends[backend]
    mu, sigma, obs = map(B.asarray, (mu, sigma, obs))

    constant = -1.0 if negative else 1.0
    ω = (obs - mu) / sigma
    prob = _norm_pdf(ω, backend=backend) / sigma
    return constant * B.log(prob)


def poisson(
    obs: "ArrayLike",
    mean: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Compute the logarithmic score for the Poisson distribution."""
    B = backends.active if backend is None else backends[backend]
    mean, obs = map(B.asarray, (mean, obs))
    prob = _pois_pdf(obs, mean, backend=backend)
    return -B.log(prob)


def t(
    obs: "ArrayLike",
    df: "ArrayLike",
    location: "ArrayLike",
    scale: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Compute the logarithmic score for the t distribution."""
    B = backends.active if backend is None else backends[backend]
    df, mu, sigma, obs = map(B.asarray, (df, location, scale, obs))
    ω = (obs - mu) / sigma
    prob = _t_pdf(ω, df, backend=backend) / sigma
    return -B.log(prob)


def tnormal(
    obs: "ArrayLike",
    location: "ArrayLike",
    scale: "ArrayLike",
    lower: "ArrayLike",
    upper: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Compute the logarithmic score for the truncated normal distribution."""
    B = backends.active if backend is None else backends[backend]
    obs, mu, sigma, lower, upper = map(B.asarray, (obs, location, scale, lower, upper))
    ω = (obs - mu) / sigma
    u = (upper - mu) / sigma
    l = (lower - mu) / sigma
    F_u = _norm_cdf(u, backend=backend)
    F_l = _norm_cdf(l, backend=backend)
    denom = F_u - F_l

    ind_out = (ω < l) | (ω > u)
    prob = _norm_pdf(ω) / sigma
    s = B.where(ind_out, float("inf"), -B.log(prob / denom))
    return s


def tt(
    obs: "ArrayLike",
    df: "ArrayLike",
    location: "ArrayLike",
    scale: "ArrayLike",
    lower: "ArrayLike",
    upper: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Compute the logarithmic score for the truncated t distribution."""
    B = backends.active if backend is None else backends[backend]
    obs, df, mu, sigma, lower, upper = map(
        B.asarray, (obs, df, location, scale, lower, upper)
    )
    ω = (obs - mu) / sigma
    u = (upper - mu) / sigma
    l = (lower - mu) / sigma
    F_u = _t_cdf(u, df, backend=backend)
    F_l = _t_cdf(l, df, backend=backend)
    denom = F_u - F_l

    ind_out = (ω < l) | (ω > u)
    prob = _t_pdf(ω, df) / sigma
    s = B.where(ind_out, float("inf"), -B.log(prob / denom))
    return s


def uniform(
    obs: "ArrayLike",
    min: "ArrayLike",
    max: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Compute the logarithmic score for the uniform distribution."""
    B = backends.active if backend is None else backends[backend]
    min, max, obs = map(B.asarray, (min, max, obs))
    ind_out = (obs > max) | (obs < min)
    prob = B.where(ind_out, 0.0, 1.0 / (max - min))
    return -B.log(prob)
