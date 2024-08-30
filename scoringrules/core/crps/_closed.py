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
    ind = (obs_std >= 0) * (obs_std <= 1)
    I_ab = B.betainc(shape1, shape2, obs_std)
    I_ab[~ind] = (obs_std[~ind] < 0) * 0 + (obs_std[~ind] > 0) * 1
    I_a1b = B.betainc(shape1 + 1, shape2, obs_std)
    I_a1b[~ind] = (obs_std[~ind] < 0) * 0 + (obs_std[~ind] > 0) * 1
    F_ab = B.minimum(B.maximum(I_ab, 0), 1)
    F_a1b = B.minimum(B.maximum(I_a1b, 0), 1)
    bet_rat = 2 * B.beta(2 * shape1, 2 * shape2) / (shape1 * B.beta(shape1, shape2)**2)
    s = obs_std * (2 * F_ab - 1) + (shape1 / (shape1 + shape2)) * ( 1 - 2 * F_a1b - bet_rat)
    return s

def binomial(
    n: "ArrayLike",
    prob: "ArrayLike",
    obs: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Compute the CRPS for the binomial distribution."""
    B = backends.active if backend is None else backends[backend]
    n, prob, obs = map(B.asarray, (n, prob, obs))
    x = list(range(0, n + 1))
    f_np = _binom_pdf(x, n, prob, backend=backend)
    F_np = _binom_cdf(x, n, prob, backend=backend) 
    s = B.sum(f_np * (B.ispositive(x - obs) - F_np + f_np / 2) * (x - obs))
    return s

def exponential(
    rate: "ArrayLike", obs: "ArrayLike", backend: "Backend" = None
) -> "Array":
    """Compute the CRPS for the exponential distribution."""
    B = backends.active if backend is None else backends[backend]
    rate, obs = map(B.asarray, (rate, obs))
    s = B.abs(obs) - (2 * _exp_cdf(obs, rate, backend=backend) / rate) + 1 / (2 * rate)
    return s

def exponentialM(
    mass: "ArrayLike", location: "ArrayLike", scale: "ArrayLike", obs: "ArrayLike", backend: "Backend" = None
) -> "Array":
    """Compute the CRPS for the standard exponential distribution with a point mass at the boundary."""
    B = backends.active if backend is None else backends[backend]
    mass, location, scale, obs = map(B.asarray, (mass, location, scale, obs))
    ω = (obs - location) / scale
    F_y = B.maximum(1 - B.exp(-ω), 0)
    s = B.abs(ω) - 2 * (1 - mass) * F_y + ((1 - mass)**2) / 2
    return s * scale

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
    s = obs * (2 * F_ab - 1) - (shape / rate) * (2 * F_ab1 - 1) - 1 / (rate * B.beta(0.5, shape))
    return s

def gev(
    shape: "ArrayLike",
    location: "ArrayLike",
    scale: "ArrayLike",
    obs: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Compute the CRPS for the GEV distribution."""
    B = backends.active if backend is None else backends[backend]
    shape, location, scale, obs = map(B.asarray, (shape, location, scale, obs))
    EULERMASCHERONI = 0.57721566490153286060651209008240243
    ω = (obs - location) / scale
    x = (1 + shape * ω)
    x[x < 0] = 0
    shape1 = shape[shape != 0]
    ω1 = ω[shape != 0]
    ω0 = ω[shape == 0]
    x1 = x[shape != 0]**(- 1 / shape1)
    c1 = 2 * B.exp(-x1)
    F0 = B.exp(-B.exp(-ω0))
    F1 = _gamma_cdf(x1, 1 - shape1, 1, backend=backend)
    s = - ω
    s[shape != 0] += - 1 / shape1 + (ω1 + 1 / shape1) * c1 + B.gamma(1 - shape1) * (2 * F1 - 2**shape1) / shape1 
    s[shape == 0] += - 2 * B.expi(B.log(F0)) + EULERMASCHERONI - B.log(2)
    return scale * s

def gpd(
    shape: "ArrayLike",
    location: "ArrayLike",
    scale: "ArrayLike",
    mass: "ArrayLike",
    obs: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Compute the CRPS for the GPD distribution."""
    B = backends.active if backend is None else backends[backend]
    shape, location, scale, mass, obs = map(B.asarray, (shape, location, scale, mass, obs))
    ω = (obs - location) / scale
    x = 1 + shape * ω
    x[x < 0] = 0
    shape1 = shape[shape != 0]
    ω0 = ω[shape == 0]
    x[shape == 0] = B.exp(-ω0)
    x[shape != 0] = x[shape != 0]**(- 1 / shape1)
    x[x > 1] = 1
    a = 1 - mass
    b = 1 - shape
    s = B.abs(obs - location) - scale * a * ((2 / b) * (1 - x**b) - a / (2 - shape))
    return s

def gtclogistic(
    location: "ArrayLike", scale: "ArrayLike", lower: "ArrayLike", upper: "ArrayLike", lmass: "ArrayLike", umass: "ArrayLike", obs: "ArrayLike", backend: "Backend" = None
) -> "Array":
    """Compute the CRPS for the generalised truncated and censored logistic distribution."""
    B = backends.active if backend is None else backends[backend]
    mu, sigma, lower, upper, lmass, umass, obs = map(B.asarray, (location, scale, lower, upper, lmass, umass, obs))
    ω = (obs - mu) / sigma
    u = (upper - mu) / sigma
    l = (lower - mu) / sigma
    z = B.minimum(B.maximum(ω, l), u)
    F_u = _logis_cdf(u, backend=backend)
    F_l = _logis_cdf(l, backend=backend)
    F_mu = _logis_cdf(-u, backend=backend)
    F_ml = _logis_cdf(-l, backend=backend)
    F_mz = _logis_cdf(-z, backend=backend)
    G_u = u * F_u + B.log(F_mu)
    G_l = l * F_l + B.log(F_ml)
    H_u = F_u - u * F_u**2 + (1 - 2 * F_u) * B.log(F_mu)
    H_l = F_l - l * F_l**2 + (1 - 2 * F_l) * B.log(F_ml)
    c = (1 - lmass - umass) / (F_u - F_l)
    s1 = B.abs(ω - z) + u * umass**2 - l * lmass**2
    s2 = c * z * ((1 - 2 * lmass) * F_u + (1 - 2 * umass) * F_l) / (1 - lmass - umass)
    s3 = c * (2 * B.log(F_mz) - 2 * G_u * umass - 2 * G_l * lmass)
    s4 = c**2 * (H_u - H_l)
    return sigma * (s1 - s2 - s3 - s4)

def gtcnormal(
    location: "ArrayLike", scale: "ArrayLike", lower: "ArrayLike", upper: "ArrayLike", lmass: "ArrayLike", umass: "ArrayLike", obs: "ArrayLike", backend: "Backend" = None
) -> "Array":
    """Compute the CRPS for the generalised truncated and censored normal distribution."""
    B = backends.active if backend is None else backends[backend]
    mu, sigma, lower, upper, lmass, umass, obs = map(B.asarray, (location, scale, lower, upper, lmass, umass, obs))
    ω = (obs - mu) / sigma
    u = (upper - mu) / sigma
    l = (lower - mu) / sigma
    z = B.minimum(B.maximum(ω, l), u)
    F_u = _norm_cdf(u, backend=backend)
    F_l = _norm_cdf(l, backend=backend)
    F_z = _norm_cdf(z, backend=backend)
    F_u2 = _norm_cdf(u * B.sqrt(2), backend=backend)
    F_l2 = _norm_cdf(l * B.sqrt(2), backend=backend)
    f_u = _norm_pdf(u, backend=backend)
    f_l = _norm_pdf(l, backend=backend)
    f_z = _norm_pdf(z, backend=backend)
    c = (1 - lmass - umass) / (F_u - F_l)
    s1 = B.abs(ω - z)
    ind = (l != -float("inf"))
    s1[ind] -= l[ind] * lmass[ind]**2
    ind = (u != float("inf"))
    s1[ind] += u[ind] * umass[ind]**2
    s2 = c * z * (2 * F_z - ((1 - 2 * lmass) * F_u + (1 - 2 * umass) * F_l) / (1 - lmass - umass))
    s3 = c * (2 * f_z - 2 * f_u * umass - 2 * f_l * lmass)
    s4 = c**2 * (F_u2 - F_l2) / B.sqrt(B.pi)
    return sigma * (s1 + s2 + s3 - s4)

def gtct(
    df: "ArrayLike", location: "ArrayLike", scale: "ArrayLike", lower: "ArrayLike", upper: "ArrayLike", lmass: "ArrayLike", umass: "ArrayLike", obs: "ArrayLike", backend: "Backend" = None
) -> "Array":
    """Compute the CRPS for the generalised truncated and censored t distribution."""
    B = backends.active if backend is None else backends[backend]
    df, mu, sigma, lower, upper, lmass, umass, obs = map(B.asarray, (df, location, scale, lower, upper, lmass, umass, obs))
    ω = (obs - mu) / sigma
    u = (upper - mu) / sigma
    l = (lower - mu) / sigma
    z = B.minimum(B.maximum(ω, l), u)
    F_u = _t_cdf(u, df, backend=backend)
    F_l = _t_cdf(l, df, backend=backend)
    F_z = _t_cdf(z, df, backend=backend)
    f_u = _t_pdf(u, df, backend=backend)
    f_l = _t_pdf(l, df, backend=backend)
    f_z = _t_pdf(z, df, backend=backend)
    G_u = - f_u * (df + u**2) / (df - 1)
    G_l = - f_l * (df + l**2) / (df - 1)
    G_z = - f_z * (df + z**2) / (df - 1)
    H_u = ((u / B.abs(u)) * B.betainc(1 / 2, df - 1 / 2, (u**2) / (df + u**2)) + 1) / 2
    H_l = ((l / B.abs(l)) * B.betainc(1 / 2, df - 1 / 2, (l**2) / (df + l**2)) + 1) / 2
    Bbar = (2 * B.sqrt(df) / (df - 1)) * B.beta(1 / 2, df - 1 / 2) / (B.beta(1 / 2, df / 2)**2)
    c = (1 - lmass - umass) / (F_u - F_l)
    s1 = B.abs(ω - z) + u * umass**2 - l * lmass**2
    s2 = c * z * (2 * F_z - ((1 - 2 * lmass) * F_u + (1 - 2 * umass) * F_l) / (1 - lmass - umass))
    s3 = 2 * c * (G_z - G_u * umass - G_l * lmass)
    s4 = c**2 * Bbar * (H_u - H_l)
    return sigma * (s1 + s2 - s3 - s4)

def hypergeometric(
    m: "ArrayLike",
    n: "ArrayLike",
    k: "ArrayLike",
    obs: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Compute the CRPS for the hypergeometric distribution."""
    B = backends.active if backend is None else backends[backend]
    m, n, k, obs = map(B.asarray, (m, n, k, obs))
    x = B.range(0, n)
    f_np = _hypergeo_pdf(x, m, n, k, backend=backend)
    F_np = _hypergeo_cdf(x, m, n, k, backend=backend)
    s = B.sum(f_np * (B.ispositive(x - obs) - F_np + f_np / 2) * (x - obs))
    return s

def laplace(
    location: "ArrayLike", scale: "ArrayLike", obs: "ArrayLike", backend: "Backend" = None
) -> "Array":
    """Compute the CRPS for the laplace distribution."""
    B = backends.active if backend is None else backends[backend]
    mu, sigma, obs = map(B.asarray, (location, scale, obs))
    ω = (obs - mu) / sigma
    return sigma * (B.abs(ω) + B.exp(-B.abs(ω)) - 3/4)

def logistic(
    mu: "ArrayLike", sigma: "ArrayLike", obs: "ArrayLike", backend: "Backend" = None
) -> "Array":
    """Compute the CRPS for the logistic distribution."""
    B = backends.active if backend is None else backends[backend]
    mu, sigma, obs = map(B.asarray, (mu, sigma, obs))
    ω = (obs - mu) / sigma
    return sigma * (ω - 2 * B.log(_logis_cdf(ω, backend=backend)) - 1)

def loglaplace(
    locationlog: "ArrayLike",
    scalelog: "ArrayLike",
    obs: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Compute the CRPS for the log-laplace distribution."""
    B = backends.active if backend is None else backends[backend]
    mulog, sigmalog, obs = map(B.asarray, (locationlog, scalelog, obs))
    F_ls = B.exp((B.log(B.abs(obs)) - mulog) / sigmalog) / 2
    F_ls[obs <= 0] = 0
    ind = obs >= B.exp(mulog)
    F_ls[ind] = 1 - B.exp( - (B.log(obs[ind]) - mulog[ind]) / sigmalog[ind]) / 2
    A = (1 - (2 * F_ls)**(1 + sigmalog)) / (1 + sigmalog)
    A[ind] = - (1 - (2 * (1 - F_ls[ind]))**(1 - sigmalog[ind])) / (1 - sigmalog[ind])
    s = obs * (2 * F_ls - 1) + B.exp(locationlog) * (A + sigmalog / (4 - sigmalog**2))
    return s

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
    s = obs * (2 * F_ms - 1) - B.exp(mulog) * b * (2 * I + sigmalog - 1)
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
    ω = (B.log(B.abs(obs)) - mulog) / sigmalog
    ex = 2 * B.exp(mulog + sigmalog**2 / 2)
    F_n = _norm_cdf(ω, backend=backend)
    F_ns = _norm_cdf(ω - sigmalog, backend=backend)
    F_c = _norm_cdf(sigmalog / B.sqrt(2), backend=backend)
    F_n[obs <= 0] = 0
    F_ns[obs <= 0] = 0
    s = obs * (2 * F_n - 1) - ex * (F_ns + F_c - 1)
    return s

def negativebinomial(
    n: "ArrayLike",
    prob: "ArrayLike",
    obs: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Compute the CRPS for the negative binomial distribution."""
    B = backends.active if backend is None else backends[backend]
    n, prob, obs = map(B.asarray, (n, prob, obs))
    F_np = _negbinom_cdf(obs, n, prob, backend=backend)
    F_n1p = _negbinom_cdf(obs - 1, n, prob, backend=backend)
    F2 = B.hypergeometric(n + 1, 1 / 2, 2, - 4 * (1 - prob) / (prob**2))
    s = obs * (2 * F_np - 1) - n * (1 - prob) * (prob * (2 * F_n1p - 1) + F2) / (prob**2)
    return s

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
    """Compute the CRPS for the poisson distribution."""
    B = backends.active if backend is None else backends[backend]
    mean, obs = map(B.asarray, (mean, obs))
    F_m = _pois_cdf(obs, mean, backend=backend)
    f_m = _pois_pdf(B.floor(obs), mean, backend=backend)
    I0 = B.mbessel0(2 * mean)
    I1 = B.mbessel1(2 * mean)
    s = (y - mean) * (2 * F_m - 1) + 2 * mean * f_m - mean * B.exp(-2 * mean) * (I0 + I1)
    return s


def logistic(
    obs: "ArrayLike", mu: "ArrayLike", sigma: "ArrayLike", backend: "Backend" = None
) -> "Array":
    """Compute the CRPS for the uniform distribution."""
    B = backends.active if backend is None else backends[backend]
    min, max, lmass, umass, obs = map(B.asarray, (min, max, lmass, umass, obs))
    ω = (obs - min) / (max - min)
    F_ω = B.minimum(B.maximum(ω, 0), 1)
    s = B.abs(ω - F_ω) + (F_ω**2) * (1 - lmass - umass)  - F_ω * (1 - 2 * lmass) + ((1 - lmass - umass)**2) / 3 + (1 - lmass) * umass
    return (max - min) * s

def t(
    df: "ArrayLike", location: "ArrayLike", scale: "ArrayLike", obs: "ArrayLike", backend: "Backend" = None
) -> "Array":
    """Compute the CRPS for the t distribution."""
    B = backends.active if backend is None else backends[backend]
    df, mu, sigma, obs = map(B.asarray, (df, location, scale, obs))
    ω = (obs - mu) / sigma
    return sigma * (ω - 2 * B.log(_logis_cdf(ω, backend=backend)) - 1)


def _is_scalar_value(x, value):
    if x.size != 1:
        return False
    return x.item() == value
