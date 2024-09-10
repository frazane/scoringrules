import typing as tp

from scoringrules.backend import backends
from scoringrules.core.stats import (
    _laplace_pdf,
    _norm_pdf,
)

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike, Backend


def laplace(
    obs: "ArrayLike",
    location: "ArrayLike",
    scale: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Compute the logarithmic score for the laplace distribution."""
    B = backends.active if backend is None else backends[backend]
    obs, mu, sigma = map(B.asarray, (obs, location, scale))
    ω = (obs - mu) / sigma
    prob = _laplace_pdf(ω, backend=backend) / sigma
    return -B.log(prob)


def loglaplace(
    obs: "ArrayLike",
    locationlog: "ArrayLike",
    scalelog: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Compute the logarithmic score for the log-laplace distribution."""
    B = backends.active if backend is None else backends[backend]
    mulog, sigmalog, obs = map(B.asarray, (locationlog, scalelog, obs))
    zind = obs <= 0.0
    obs = B.where(zind, B.nan, obs)
    ω = (B.log(obs) - mulog) / sigmalog
    prob = _laplace_pdf(ω, backend=backend) / sigmalog
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
    prob = _norm_pdf(ω) / sigma
    return constant * B.log(prob)
