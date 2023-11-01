import typing as tp

from scoringrules.backend import backends
from scoringrules.core.stats import _norm_pdf

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike


def normal(
    mu: "ArrayLike",
    sigma: "ArrayLike",
    obs: "ArrayLike",
    negative: bool = True,
    backend: str | None = None,
) -> "Array":
    """Compute the logarithmic score for the normal distribution."""
    B = backends.active if backend is None else backends[backend]
    ciao = B.asarray(mu)
    print(ciao)
    mu, sigma, obs = map(B.asarray, (mu, sigma, obs))

    constant = -1.0 if negative else 1.0
    ω = (obs - mu) / sigma
    prob = _norm_pdf(ω) / sigma
    return constant * B.log(prob)
