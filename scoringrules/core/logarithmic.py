import typing as tp

from scoringrules.backend import backends
from scoringrules.core.stats import (
    _exp_pdf,
    _norm_pdf,
)

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike, Backend


def beta(
    obs: "ArrayLike",
    a: "ArrayLike",
    b: "ArrayLike",
    lower: "ArrayLike" = 0.0,
    upper: "ArrayLike" = 1.0,
    backend: "Backend" = None,
) -> "Array":
    """Compute the logarithmic score for the beta distribution."""
    B = backends.active if backend is None else backends[backend]
    a, b, lower, upper, obs = map(B.asarray, (a, b, lower, upper, obs))
    con = B.beta(a, b)
    scale = upper - lower
    obs = (obs - lower) / scale
    ind_out = (obs < 0.0) | (obs > 1.0)
    obs = B.where(ind_out, B.nan, obs)
    prob = (obs ** (a - 1)) * ((1 - obs) ** (b - 1)) / con
    s = B.where(ind_out, float("inf"), -B.log(prob / scale))
    return s


def exponential2(
    obs: "ArrayLike",
    location: "ArrayLike",
    scale: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Compute the logarithmic score for the exponential distribution with location and scale parameters."""
    B = backends.active if backend is None else backends[backend]
    obs, location, scale = map(B.asarray, (obs, location, scale))
    obs -= location
    rate = 1 / scale

    zind = obs < 0.0
    obs = B.where(zind, B.nan, obs)
    prob = _exp_pdf(obs, rate, backend=backend)
    s = B.where(zind, float("inf"), -B.log(prob))
    return s


def twopexponential(
    obs: "ArrayLike",
    scale1: "ArrayLike",
    scale2: "ArrayLike",
    location: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Compute the logarithmic score for the two-piece exponential distribution."""
    B = backends.active if backend is None else backends[backend]
    scale1, scale2, location, obs = map(B.asarray, (scale1, scale2, location, obs))
    obs = obs - location
    scale = B.where(obs < 0.0, scale1, scale2)
    rate = 1 / scale
    con = scale / (scale1 + scale2)
    prob = _exp_pdf(B.abs(obs), rate, backend=backend) * con
    return -B.log(prob)


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
    prob = _norm_pdf(ω) / sigma
    return constant * B.log(prob)


def twopnormal(
    obs: "ArrayLike",
    scale1: "ArrayLike",
    scale2: "ArrayLike",
    location: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Compute the logarithmic score for the two-piece normal distribution."""
    B = backends.active if backend is None else backends[backend]
    scale1, scale2, location, obs = map(B.asarray, (scale1, scale2, location, obs))
    obs = obs - location
    scale = B.where(obs < 0.0, scale1, scale2)
    obs = obs / scale
    con = 2 * scale / (scale1 + scale2)
    prob = _norm_pdf(obs, backend=backend) * con / scale
    return -B.log(prob)
