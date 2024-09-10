import typing as tp

from scoringrules.backend import backends
from scoringrules.core.stats import (
    _negbinom_pdf,
    _norm_pdf,
)

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike, Backend


def negbinom(
    obs: "ArrayLike",
    n: "ArrayLike",
    prob: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Compute the logarithmic score for the negative binomial distribution."""
    B = backends.active if backend is None else backends[backend]
    n, prob, obs = map(B.asarray, (n, prob, obs))
    zind = (obs < 0.0) | (B.floor(obs) != obs)
    obs = B.where(zind, B.nan, obs)
    prob = _negbinom_pdf(obs, n, prob, backend=backend)
    s = B.where(zind, float("inf"), -B.log(prob))
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
