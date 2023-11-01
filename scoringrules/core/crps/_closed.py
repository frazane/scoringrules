import typing as tp

from scoringrules.backend import backends
from scoringrules.core.stats import _logis_cdf, _norm_cdf, _norm_pdf

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike


def normal(
    mu: "ArrayLike", sigma: "ArrayLike", obs: "ArrayLike", backend: str | None = None
) -> "Array":
    """Compute the CRPS for the normal distribution."""
    B = backends.active if backend is None else backends[backend]
    mu, sigma, obs = map(B.asarray, (mu, sigma, obs))
    ω = (obs - mu) / sigma
    return sigma * (ω * (2 * _norm_cdf(ω) - 1) + 2 * _norm_pdf(ω) - 1 / B.sqrt(B.pi))


def lognormal(
    mulog: "ArrayLike",
    sigmalog: "ArrayLike",
    obs: "ArrayLike",
    backend: str | None = None,
) -> "Array":
    """Compute the CRPS for the lognormal distribution."""
    B = backends.active if backend is None else backends[backend]
    mulog, sigmalog, obs = map(B.asarray, (mulog, sigmalog, obs))
    ω = (B.log(obs) - mulog) / sigmalog
    ex = 2 * B.exp(mulog + sigmalog**2 / 2)
    return obs * (2.0 * _norm_cdf(ω) - 1) - ex * (
        _norm_cdf(ω - sigmalog) + _norm_cdf(sigmalog / B.sqrt(B.asarray(2.0))) - 1
    )


def logistic(
    mu: "ArrayLike", sigma: "ArrayLike", obs: "ArrayLike", backend: str | None = None
) -> "Array":
    """Compute the CRPS for the normal distribution."""
    B = backends.active if backend is None else backends[backend]
    mu, sigma, obs = map(B.asarray, (mu, sigma, obs))
    ω = (obs - mu) / sigma
    return sigma * (ω - 2 * B.log(_logis_cdf(ω)) - 1)
