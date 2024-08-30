import typing as tp

from scoringrules.backend import backends
from scoringrules.core.stats import (
    _binom_cdf,
    _binom_pdf,
    _exp_cdf,
    _gamma_cdf,
    _logis_cdf,
    _norm_cdf,
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
    """Compute the CRPS for the beta distribution."""
    B = backends.active if backend is None else backends[backend]
    obs, a, b, lower, upper = map(B.asarray, (obs, a, b, lower, upper))

    if _is_scalar_value(lower, 0.0) and _is_scalar_value(upper, 1.0):
        special_limits = False
    else:
        if B.any(lower >= upper):
            raise ValueError("lower must be less than upper")
        special_limits = True

    if special_limits:
        obs = (obs - lower) / (upper - lower)

    I_ab = B.betainc(a, b, obs)
    I_a1b = B.betainc(a + 1, b, obs)
    F_ab = B.minimum(B.maximum(I_ab, 0), 1)
    F_a1b = B.minimum(B.maximum(I_a1b, 0), 1)
    bet_rat = 2 * B.beta(2 * a, 2 * b) / (a * B.beta(a, b) ** 2)
    s = obs * (2 * F_ab - 1) + (a / (a + b)) * (1 - 2 * F_a1b - bet_rat)

    if special_limits:
        s = s * (upper - lower)

    return s


def binomial(
    obs: "ArrayLike",
    n: "ArrayLike",
    prob: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Compute the CRPS for the binomial distribution.

    Note
    ----
    This is a bit of a hacky implementation, due to how the arrays
    must be broadcasted, but it should work for now.
    """
    B = backends.active if backend is None else backends[backend]
    obs, n, prob = map(B.asarray, (obs, n, prob))
    ones_like_n = 0.0 * n + 1

    def _inner(params):
        obs, n, prob = params
        x = B.arange(0, n + 1)
        w = _binom_pdf(x, n, prob)
        a = _binom_cdf(x, n, prob) - 0.5 * w
        s = 2 * B.sum(w * ((obs < x) - a) * (x - obs))
        return s

    # if n is a scalar, then if needed we must broadcast k and p to the same shape as n
    # TODO: implement B.broadcast() for backends
    if n.size == 1:
        x = B.arange(0, n + 1)
        need_broadcast = not (obs.size == 1 and prob.size == 1)

        if need_broadcast:
            obs = obs[:, None] if obs.size > 1 else obs[None]
            prob = prob[:, None] if prob.size > 1 else prob[None]
            x = x[None]
            x = x * ones_like_n
            prob = prob * ones_like_n
            obs = obs * ones_like_n

        w = _binom_pdf(x, n, prob)
        a = _binom_cdf(x, n, prob) - 0.5 * w
        s = 2 * B.sum(
            w * ((obs < x) - a) * (x - obs), axis=-1 if need_broadcast else None
        )

    # otherwise, since x would have variable sizes, we must apply the function along the axis
    else:
        obs = obs * ones_like_n if obs.size == 1 else obs
        prob = prob * ones_like_n if prob.size == 1 else prob

        # option 1: in a loop
        s = B.stack(
            [_inner(params) for params in zip(obs, n, prob, strict=True)], axis=-1
        )

        # option 2: apply_along_axis (does not work with JAX)
        # s = B.apply_along_axis(_inner, B.stack((obs, n, prob), axis=-1), -1)

    return s


def exponential(
    obs: "ArrayLike", rate: "ArrayLike", backend: "Backend" = None
) -> "Array":
    """Compute the CRPS for the exponential distribution."""
    B = backends.active if backend is None else backends[backend]
    rate, obs = map(B.asarray, (rate, obs))
    s = B.abs(obs) - (2 * _exp_cdf(obs, rate, backend=backend) / rate) + 1 / (2 * rate)
    return s


def exponentialM(
    obs: "ArrayLike",
    mass: "ArrayLike",
    location: "ArrayLike",
    scale: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Compute the CRPS for the standard exponential distribution with a point mass at the boundary."""
    B = backends.active if backend is None else backends[backend]
    obs, location, scale, mass = map(B.asarray, (obs, location, scale, mass))

    if not _is_scalar_value(location, 0.0):
        obs -= location

    a = 1.0 if _is_scalar_value(mass, 0.0) else 1 - mass
    s = B.abs(obs)

    if _is_scalar_value(scale, 1.0):
        s -= a * (2 * _exp_cdf(obs, 1.0, backend=backend) - 0.5 * a)
    else:
        s -= scale * a * (2 * _exp_cdf(obs, 1 / scale, backend=backend) - 0.5 * a)

    return s


def gamma(
    obs: "ArrayLike",
    shape: "ArrayLike",
    rate: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Compute the CRPS for the gamma distribution."""
    B = backends.active if backend is None else backends[backend]
    obs, shape, rate = map(B.asarray, (obs, shape, rate))
    F_ab = _gamma_cdf(obs, shape, rate, backend=backend)
    F_ab1 = _gamma_cdf(obs, shape + 1, rate, backend=backend)
    s = (
        obs * (2 * F_ab - 1)
        - (shape / rate) * (2 * F_ab1 - 1)
        - 1 / (rate * B.beta(B.asarray(0.5), shape))
    )
    return s


def normal(
    obs: "ArrayLike", mu: "ArrayLike", sigma: "ArrayLike", backend: "Backend" = None
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


def lognormal(
    obs: "ArrayLike",
    mulog: "ArrayLike",
    sigmalog: "ArrayLike",
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


def logistic(
    obs: "ArrayLike", mu: "ArrayLike", sigma: "ArrayLike", backend: "Backend" = None
) -> "Array":
    """Compute the CRPS for the logistic distribution."""
    B = backends.active if backend is None else backends[backend]
    mu, sigma, obs = map(B.asarray, (mu, sigma, obs))
    ω = (obs - mu) / sigma
    return sigma * (ω - 2 * B.log(_logis_cdf(ω, backend=backend)) - 1)


def _is_scalar_value(x, value):
    if x.size != 1:
        return False
    return x.item() == value
