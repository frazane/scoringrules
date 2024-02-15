import typing as tp

from scoringrules.backend import backends
from scoringrules.core.stats import (
    _logis_cdf,
    _norm_cdf,
    _norm_pdf,
    _exp_cdf,
    _gamma_cdf,
    _pois_cdf,
    _pois_pdf,
    _t_cdf,
    _t_pdf
)

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike, Backend

def beta(
    shape1: "ArrayLike",
    shape2: "ArrayLike",
    lower: "ArrayLike",
    upper: "ArrayLike",
    obs: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Compute the CRPS for the beta distribution."""
    B = backends.active if backend is None else backends[backend]
    shape1, shape2, lower, upper, obs = map(B.asarray, (shape1, shape2, lower, upper, obs))
    obs_std = (obs - lower) / (upper - lower)
    I_ab = B.betainc(shape1, shape2, obs_std)
    I_a1b = B.betainc(shape1 + 1, shape2, obs_std)
    F_ab = B.minimum(B.maximum(I_ab, 0), 1)
    F_a1b = B.minimum(B.maximum(I_a1b, 0), 1)
    bet_rat = 2 * B.beta(2 * shape1, 2 * shape2) / (shape1 * B.beta(shape1, shape2)**2)
    s = obs_std * (2 * F_ab - 1) + (shape1 / (shape1 + shape2)) * ( 1 - 2 * F_a1b - bet_rat)
    return s


def exponential(
    rate: "ArrayLike", obs: "ArrayLike", backend: "Backend" = None
) -> "Array":
    """Compute the CRPS for the exponential distribution."""
    B = backends.active if backend is None else backends[backend]
    rate, obs = map(B.asarray, (rate, obs))
    s = B.abs(obs) - (2 * _exp_cdf(obs, rate, backend=backend) / rate) + 1 / (2 * rate)
    return s

def gamma(
    shape: "ArrayLike",
    rate: "ArrayLike",
    scale: "ArrayLike",
    obs: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Compute the CRPS for the gamma distribution."""
    B = backends.active if backend is None else backends[backend]
    shape, rate, scale, obs = map(B.asarray, (shape, rate, scale, obs))
    F_ab = _gamma_cdf(obs, shape, rate, backend=backend)
    F_ab1 = _gamma_cdf(obs, shape + 1, rate, backend=backend)
    s = obs * (2 * F_ab - 1) + (shape / rate) * (2 * F_ab1 - 1) - 1 / (rate * B.beta(0.5, shape))
    return s

def laplace(
    location: "ArrayLike", scale: "ArrayLike", obs: "ArrayLike", backend: "Backend" = None
) -> "Array":
    """Compute the CRPS for the laplace distribution."""
    B = backends.active if backend is None else backends[backend]
    mu, sigma, obs = map(B.asarray, (location, scale, obs))
    ω = (obs - mu) / sigma
    return sigma * (B.abs(ω) + B.exp(-ω) - 3/4)

def logistic(
    mu: "ArrayLike", sigma: "ArrayLike", obs: "ArrayLike", backend: "Backend" = None
) -> "Array":
    """Compute the CRPS for the logistic distribution."""
    B = backends.active if backend is None else backends[backend]
    mu, sigma, obs = map(B.asarray, (mu, sigma, obs))
    ω = (obs - mu) / sigma
    return sigma * (ω - 2 * B.log(_logis_cdf(ω, backend=backend)) - 1)

def loglogistic(
    mulog: "ArrayLike",
    sigmalog: "ArrayLike",
    obs: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Compute the CRPS for the log-logistic distribution."""
    B = backends.active if backend is None else backends[backend]
    mulog, sigmalog, obs = map(B.asarray, (mulog, sigmalog, obs))
    F_ms = 1 / (1 + B.exp( - (B.log(obs) - mulog) / sigmalog))
    b = B.beta(1 + sigmalog, 1 - sigmalog)
    I = B.betainc(1 + sigmalog, 1 - sigmalog, F_ms)
    s = obs * (2 * F_ms - 1) - B.exp(mulog) * b * (2*I + sigmalog - 1)
    return s

def lognormal(
    mulog: "ArrayLike",
    sigmalog: "ArrayLike",
    obs: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Compute the CRPS for the lognormal distribution."""
    B = backends.active if backend is None else backends[backend]
    mulog, sigmalog, obs = map(B.asarray, (mulog, sigmalog, obs))
    ω = (B.log(obs) - mulog) / sigmalog
    ex = 2 * B.exp(mulog + sigmalog**2 / 2)
    return obs * (2.0 * _norm_cdf(ω, backend=backend) - 1) - ex * (
        _norm_cdf(ω - sigmalog, backend=backend)
        + _norm_cdf(sigmalog / B.sqrt(B.asarray(2.0)), backend=backend)
        - 1
    )

def normal(
    mu: "ArrayLike", sigma: "ArrayLike", obs: "ArrayLike", backend: "Backend" = None
) -> "Array":
    """Compute the CRPS for the normal distribution."""
    B = backends.active if backend is None else backends[backend]
    mu, sigma, obs = map(B.asarray, (mu, sigma, obs))
    ω = (obs - mu) / sigma
    return sigma * (
        ω * (2.0 * _norm_cdf(ω, backend=backend) - 1.0)
        + 2.0 * _norm_pdf(ω, backend=backend)
        - 1.0 / B.sqrt(B.pi)
    )

def poisson(
    mean: "ArrayLike", obs: "ArrayLike", backend: "Backend" = None
) -> "Array":
    """Compute the CRPS for the poisson distribution."""
    B = backends.active if backend is None else backends[backend]
    mean, obs = map(B.asarray, (mean, obs))
    F_m = _pois_cdf(obs, mean, backend=backend)
    f_m = _pois_pdf(B.floor(obs), mean, backend=backend)
    I0 = B.mbessel0(2 * mean)
    I1 = B.mbessel1(2 * mean)
    s = (y - mean) * (2 * F_m - 1) + 2 * mean * f_m - mean * B.exp(-2 * mean) * (I0 + I1)
    return s

def uniform(
    min: "ArrayLike", max: "ArrayLike", lmass: "ArrayLike", umass: "ArrayLike", obs: "ArrayLike", backend: "Backend" = None
) -> "Array":
    """Compute the CRPS for the uniform distribution."""
    B = backends.active if backend is None else backends[backend]
    min, max, lmass, umass, obs = map(B.asarray, (min, max, lmass, umass, obs))
    ω = (obs - min) / (max - min)
    F_ω = B.minimum(B.maximum(ω, 0), 1)
    return B.abs(ω - F_ω) + (F_ω**2) * (1 - lmass - umass)  - F_ω * (1 - 2 * lmass) + ((1 - lmass - umass)**2) / 3 + (1 - lmass) * umass

def t(
    df: "ArrayLike", location: "ArrayLike", scale: "ArrayLike", obs: "ArrayLike", backend: "Backend" = None
) -> "Array":
    """Compute the CRPS for the t distribution."""
    B = backends.active if backend is None else backends[backend]
    df, mu, sigma, obs = map(B.asarray, (df, location, scale, obs))
    ω = (obs - mu) / sigma
    F_nu = _t_cdf(ω)
    f_nu = _t_pdf(ω)
    s = ω * (2 * F_nu - 1) + 2 * f_nu * (df + ω**2) / (df - 1) - (2 * B.sqrt(df) * B.beta(1 / 2, df - 1)) / ((df - 1) * B.beta(1 / 2, df / 2)**2)
    return sigma * s

def tpexponential(
    scale1: "ArrayLike", scale2: "ArrayLike", location: "ArrayLike", obs: "ArrayLike", backend: "Backend" = None
) -> "Array":
    """Compute the CRPS for the two-piece exponential distribution."""
    B = backends.active if backend is None else backends[backend]
    scale1, scale2, location, obs = map(B.asarray, (scale1, scale2, location, obs))
    ω = (obs - location)
    scalei = scale2
    if ω < 0: scalei = scale1
    s = B.abs(ω) + 2 * (scalei**2) *B.exp( - B.abs(ω) / scalei) / (scale1 + scale2) - 2 * (scalei**2) / (scale1 + scale2) + (scale1**3 + scale2**3) / (2 * (scale1 + scale2)**2)
    return s