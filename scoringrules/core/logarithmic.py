import typing as tp

from scoringrules.core.stats_xp import (
    _binom_pdf,
    _exp_pdf,
    _gamma_pdf,
    _gev_pdf,
    _gpd_pdf,
    _hypergeo_pdf,
    _laplace_pdf,
    _logis_pdf,
    _logis_cdf,
    _negbinom_pdf,
    _norm_pdf,
    _norm_cdf,
    _pois_pdf,
    _t_pdf,
    _t_cdf,
)

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike


def clogs_ensemble(
    obs: "ArrayLike",
    fct: "Array",
    a: "ArrayLike",
    b: "ArrayLike",
    bw: "ArrayLike",
    cens: bool = True,
    *,
    xp,
) -> "Array":
    """Compute the conditional or censored likelihood score for an ensemble forecast."""
    a, b, bw, fct, obs = map(xp.asarray, (a, b, bw, fct, obs))

    z = (obs[..., None] - fct) / bw
    a_z = (a - fct) / bw
    b_z = (b - fct) / bw

    dens = _norm_pdf(z, xp=xp) / bw
    dens = xp.mean(dens, axis=-1)
    s1 = -xp.log(dens)

    F_a = _norm_cdf(a_z, xp=xp)
    F_a = xp.mean(F_a, axis=-1)
    F_b = _norm_cdf(b_z, xp=xp)
    F_b = xp.mean(F_b, axis=-1)
    cons = F_b - F_a

    w_ind = (obs > a) & (obs < b)
    if cens:
        s2 = -xp.log(1 - cons)
        s = xp.where(w_ind, s1, s2)
    else:
        s2 = xp.log(cons)
        s = xp.where(w_ind, s1 + s2, 0.0)

    return s


def beta(
    obs: "ArrayLike",
    a: "ArrayLike",
    b: "ArrayLike",
    lower: "ArrayLike" = 0.0,
    upper: "ArrayLike" = 1.0,
    *,
    xp,
) -> "Array":
    """Compute the logarithmic score for the beta distribution."""
    a, b, lower, upper, obs = map(xp.asarray, (a, b, lower, upper, obs))
    con = xp.beta(a, b)
    scale = upper - lower
    obs = (obs - lower) / scale
    ind_out = (obs < 0.0) | (obs > 1.0)
    obs = xp.where(ind_out, xp.nan, obs)
    prob = (obs ** (a - 1)) * ((1 - obs) ** (b - 1)) / con
    s = xp.where(ind_out, float("inf"), -xp.log(prob / scale))
    return s


def binomial(
    obs: "ArrayLike",
    n: "ArrayLike",
    prob: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the logarithmic score for the binomial distribution."""
    obs, n, prob = map(xp.asarray, (obs, n, prob))
    zind = obs < 0.0
    obs = xp.where(zind, xp.nan, obs)
    prob = _binom_pdf(obs, n, prob, xp=xp)
    s = xp.where(zind, float("inf"), -xp.log(prob))
    return s


def exponential(obs: "ArrayLike", rate: "ArrayLike", *, xp) -> "Array":
    """Compute the logarithmic score for the exponential distribution."""
    rate, obs = map(xp.asarray, (rate, obs))
    zind = obs < 0.0
    obs = xp.where(zind, xp.nan, obs)
    prob = _exp_pdf(obs, rate, xp=xp)
    s = xp.where(zind, float("inf"), -xp.log(prob))
    return s


def exponential2(
    obs: "ArrayLike",
    location: "ArrayLike",
    scale: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the logarithmic score for the exponential distribution with location and scale parameters."""
    obs, location, scale = map(xp.asarray, (obs, location, scale))
    obs = obs - location  # not in place: obs may be the caller's array (#153)
    rate = 1 / scale

    zind = obs < 0.0
    obs = xp.where(zind, xp.nan, obs)
    prob = _exp_pdf(obs, rate, xp=xp)
    s = xp.where(zind, float("inf"), -xp.log(prob))
    return s


def twopexponential(
    obs: "ArrayLike",
    scale1: "ArrayLike",
    scale2: "ArrayLike",
    location: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the logarithmic score for the two-piece exponential distribution."""
    scale1, scale2, location, obs = map(xp.asarray, (scale1, scale2, location, obs))
    obs = obs - location
    scale = xp.where(obs < 0.0, scale1, scale2)
    rate = 1 / scale
    con = scale / (scale1 + scale2)
    prob = _exp_pdf(xp.abs(obs), rate, xp=xp) * con
    return -xp.log(prob)


def gamma(
    obs: "ArrayLike",
    shape: "ArrayLike",
    rate: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the logarithmic score for the gamma distribution."""
    obs, shape, rate = map(xp.asarray, (obs, shape, rate))
    prob = _gamma_pdf(obs, shape, rate, xp=xp)
    return -xp.log(prob)


def gev(
    obs: "ArrayLike",
    shape: "ArrayLike",
    location: "ArrayLike",
    scale: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the logarithmic score for the GEV distribution."""
    shape, location, scale, obs = map(xp.asarray, (shape, location, scale, obs))
    ω = (obs - location) / scale
    prob = _gev_pdf(ω, shape, xp=xp)
    prob0 = prob == 0.0
    prob = xp.where(prob0, xp.nan, prob / scale)
    s = xp.where(prob0, float("inf"), -xp.log(prob))
    return s


def gpd(
    obs: "ArrayLike",
    shape: "ArrayLike",
    location: "ArrayLike",
    scale: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the logarithmic score for the GPD distribution."""
    shape, location, scale, obs = map(xp.asarray, (shape, location, scale, obs))
    ω = (obs - location) / scale
    prob = _gpd_pdf(ω, shape, xp=xp)
    prob0 = prob == 0.0
    prob = xp.where(prob0, xp.nan, prob / scale)
    s = xp.where(prob0, float("inf"), -xp.log(prob))
    return s


def hypergeometric(
    obs: "ArrayLike",
    m: "ArrayLike",
    n: "ArrayLike",
    k: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the logarithmic score for the hypergeometric distribution."""
    obs, m, n, k = map(xp.asarray, (obs, m, n, k))
    M = m + n
    N = k
    prob = _hypergeo_pdf(obs, M, m, N, xp=xp)
    return -xp.log(prob)


def laplace(
    obs: "ArrayLike",
    location: "ArrayLike",
    scale: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the logarithmic score for the laplace distribution."""
    obs, mu, sigma = map(xp.asarray, (obs, location, scale))
    ω = (obs - mu) / sigma
    prob = _laplace_pdf(ω, xp=xp) / sigma
    return -xp.log(prob)


def loglaplace(
    obs: "ArrayLike",
    locationlog: "ArrayLike",
    scalelog: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the logarithmic score for the log-laplace distribution."""
    mulog, sigmalog, obs = map(xp.asarray, (locationlog, scalelog, obs))
    zind = obs <= 0.0
    obs = xp.where(zind, xp.nan, obs)
    ω = (xp.log(obs) - mulog) / sigmalog
    prob = _laplace_pdf(ω, xp=xp) / sigmalog
    s = xp.where(zind, float("inf"), -xp.log(prob / obs))
    return s


def logistic(
    obs: "ArrayLike",
    mu: "ArrayLike",
    sigma: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the logarithmic score for the logistic distribution."""
    mu, sigma, obs = map(xp.asarray, (mu, sigma, obs))
    ω = (obs - mu) / sigma
    prob = _logis_pdf(ω, xp=xp) / sigma
    return -xp.log(prob)


def loglogistic(
    obs: "ArrayLike",
    mulog: "ArrayLike",
    sigmalog: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the logarithmic score for the log-logistic distribution."""
    mulog, sigmalog, obs = map(xp.asarray, (mulog, sigmalog, obs))
    zind = obs <= 0.0
    obs = xp.where(zind, xp.nan, obs)
    ω = (xp.log(obs) - mulog) / sigmalog
    prob = _logis_pdf(ω, xp=xp) / sigmalog
    s = xp.where(zind, float("inf"), -xp.log(prob / obs))
    return s


def lognormal(
    obs: "ArrayLike",
    mulog: "ArrayLike",
    sigmalog: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the logarithmic score for the log-normal distribution."""
    mulog, sigmalog, obs = map(xp.asarray, (mulog, sigmalog, obs))
    zind = obs <= 0.0
    obs = xp.where(zind, xp.nan, obs)
    ω = (xp.log(obs) - mulog) / sigmalog
    prob = _norm_pdf(ω, xp=xp) / sigmalog
    s = xp.where(zind, float("inf"), -xp.log(prob / obs))
    return s


def mixnorm(
    obs: "ArrayLike",
    m: "ArrayLike",
    s: "ArrayLike",
    w: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the logarithmic score for a mixture of normal distributions."""
    m, s, w, obs = map(xp.asarray, (m, s, w, obs))

    z = (obs[..., None] - m) / s
    prob = _norm_pdf(z, xp=xp) / s
    prob = xp.sum(w * prob, axis=-1)
    return -xp.log(prob)


def negbinom(
    obs: "ArrayLike",
    n: "ArrayLike",
    prob: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the logarithmic score for the negative binomial distribution."""
    n, prob, obs = map(xp.asarray, (n, prob, obs))
    zind = (obs < 0.0) | (xp.floor(obs) != obs)
    obs = xp.where(zind, xp.nan, obs)
    prob = _negbinom_pdf(obs, n, prob, xp=xp)
    s = xp.where(zind, float("inf"), -xp.log(prob))
    return s


def normal(
    obs: "ArrayLike",
    mu: "ArrayLike",
    sigma: "ArrayLike",
    negative: bool = True,
    *,
    xp,
) -> "Array":
    """Compute the logarithmic score for the normal distribution."""
    mu, sigma, obs = map(xp.asarray, (mu, sigma, obs))

    constant = -1.0 if negative else 1.0
    ω = (obs - mu) / sigma
    prob = _norm_pdf(ω, xp=xp) / sigma
    return constant * xp.log(prob)


def twopnormal(
    obs: "ArrayLike",
    scale1: "ArrayLike",
    scale2: "ArrayLike",
    location: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the logarithmic score for the two-piece normal distribution."""
    scale1, scale2, location, obs = map(xp.asarray, (scale1, scale2, location, obs))
    obs = obs - location
    scale = xp.where(obs < 0.0, scale1, scale2)
    obs = obs / scale
    con = 2 * scale / (scale1 + scale2)
    prob = _norm_pdf(obs, xp=xp) * con / scale
    return -xp.log(prob)


def poisson(
    obs: "ArrayLike",
    mean: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the logarithmic score for the Poisson distribution."""
    mean, obs = map(xp.asarray, (mean, obs))
    prob = _pois_pdf(obs, mean, xp=xp)
    return -xp.log(prob)


def t(
    obs: "ArrayLike",
    df: "ArrayLike",
    location: "ArrayLike",
    scale: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the logarithmic score for the t distribution."""
    df, mu, sigma, obs = map(xp.asarray, (df, location, scale, obs))
    ω = (obs - mu) / sigma
    prob = _t_pdf(ω, df, xp=xp) / sigma
    return -xp.log(prob)


def tlogistic(
    obs: "ArrayLike",
    location: "ArrayLike",
    scale: "ArrayLike",
    lower: "ArrayLike",
    upper: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the logarithmic score for the truncated logistic distribution."""
    obs, mu, sigma, lower, upper = map(xp.asarray, (obs, location, scale, lower, upper))
    ω = (obs - mu) / sigma
    u = (upper - mu) / sigma
    l = (lower - mu) / sigma
    F_u = _logis_cdf(u, xp=xp)
    F_l = _logis_cdf(l, xp=xp)
    denom = F_u - F_l

    ind_out = (ω < l) | (ω > u)
    prob = _logis_pdf(ω, xp=xp) / sigma
    s = xp.where(ind_out, float("inf"), -xp.log(prob / denom))
    return s


def tnormal(
    obs: "ArrayLike",
    location: "ArrayLike",
    scale: "ArrayLike",
    lower: "ArrayLike",
    upper: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the logarithmic score for the truncated normal distribution."""
    obs, mu, sigma, lower, upper = map(xp.asarray, (obs, location, scale, lower, upper))
    ω = (obs - mu) / sigma
    u = (upper - mu) / sigma
    l = (lower - mu) / sigma
    F_u = _norm_cdf(u, xp=xp)
    F_l = _norm_cdf(l, xp=xp)
    denom = F_u - F_l

    ind_out = (ω < l) | (ω > u)
    prob = _norm_pdf(ω, xp=xp) / sigma
    s = xp.where(ind_out, float("inf"), -xp.log(prob / denom))
    return s


def tt(
    obs: "ArrayLike",
    df: "ArrayLike",
    location: "ArrayLike",
    scale: "ArrayLike",
    lower: "ArrayLike",
    upper: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the logarithmic score for the truncated t distribution."""
    obs, df, mu, sigma, lower, upper = map(
        xp.asarray, (obs, df, location, scale, lower, upper)
    )
    ω = (obs - mu) / sigma
    u = (upper - mu) / sigma
    l = (lower - mu) / sigma
    F_u = _t_cdf(u, df, xp=xp)
    F_l = _t_cdf(l, df, xp=xp)
    denom = F_u - F_l

    ind_out = (ω < l) | (ω > u)
    prob = _t_pdf(ω, df, xp=xp) / sigma
    s = xp.where(ind_out, float("inf"), -xp.log(prob / denom))
    return s


def uniform(
    obs: "ArrayLike",
    min: "ArrayLike",
    max: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the logarithmic score for the uniform distribution."""
    min, max, obs = map(xp.asarray, (min, max, obs))
    ind_out = (obs > max) | (obs < min)
    prob = xp.where(ind_out, 0.0, 1.0 / (max - min))
    return -xp.log(prob)
