import typing as tp

from scoringrules.backend import backends
from scoringrules.core.stats import (
    _logis_pdf,
    _logis_cdf,
    _norm_pdf,
    _norm_cdf,
    _t_pdf,
    _t_cdf,
)

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike, Backend


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
