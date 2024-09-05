import typing as tp

from scoringrules.backend import backends
from scoringrules.core.stats import (
    _binom_cdf,
    _binom_pdf,
    _exp_cdf,
    _gamma_cdf,
    _gev_cdf,
    _gpd_cdf,
    _hypergeo_cdf,
    _hypergeo_pdf,
    _logis_cdf,
    _norm_cdf,
    _norm_pdf,
    _t_cdf,
    _t_pdf,
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


EULERMASCHERONI = 0.57721566490153286060651209008240243


def gev(
    obs: "ArrayLike",
    shape: "ArrayLike",
    location: "ArrayLike",
    scale: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Compute the CRPS for the GEV distribution."""
    B = backends.active if backend is None else backends[backend]
    obs, shape, location, scale = map(B.asarray, (obs, shape, location, scale))

    obs = (obs - location) / scale
    # if not _is_scalar_value(location, 0.0):
    # obs -= location

    # if not _is_scalar_value(scale, 1.0):
    # obs /= scale

    def _gev_adjust_fn(s, xi, f_xi):
        res = B.nan * s
        p_xi = xi > 0
        n_xi = xi < 0
        n_inv_xi = -1 / xi

        gen_res = n_inv_xi * f_xi + B.gammauinc(1 - xi, -B.log(f_xi)) / xi

        res = B.where(p_xi & (s <= n_inv_xi), 0, res)
        res = B.where(p_xi & (s > n_inv_xi), gen_res, res)

        res = B.where(n_xi & (s < n_inv_xi), gen_res, res)
        res = B.where(n_xi & (s >= n_inv_xi), n_inv_xi + B.gamma(1 - xi) / xi, res)

        return res

    F_xi = _gev_cdf(obs, shape, backend=backend)
    zero_shape = shape == 0.0
    shape = B.where(~zero_shape, shape, B.nan)
    G_xi = _gev_adjust_fn(obs, shape, F_xi)

    out = B.where(
        zero_shape,
        -obs - 2 * B.expi(B.log(F_xi)) + EULERMASCHERONI - B.log(2),
        obs * (2 * F_xi - 1)
        - 2 * G_xi
        - (1 - (2 - 2**shape) * B.gamma(1 - shape)) / shape,
    )

    out = out * scale

    return float(out) if out.size == 1 else out


def gpd(
    obs: "ArrayLike",
    shape: "ArrayLike",
    location: "ArrayLike",
    scale: "ArrayLike",
    mass: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Compute the CRPS for the GPD distribution."""
    B = backends.active if backend is None else backends[backend]
    shape, location, scale, mass, obs = map(
        B.asarray, (shape, location, scale, mass, obs)
    )
    shape = B.where(shape < 1.0, shape, B.nan)
    mass = B.where((mass >= 0.0) & (mass <= 1.0), mass, B.nan)
    ω = (obs - location) / scale
    F_xi = _gpd_cdf(ω, shape, backend=backend)
    s = (
        B.abs(ω)
        - 2 * (1 - mass) * (1 - (1 - F_xi) ** (1 - shape)) / (1 - shape)
        + ((1 - mass) ** 2) / (2 - shape)
    )
    return scale * s


def gtclogistic(
    obs: "ArrayLike", location: "ArrayLike", scale: "ArrayLike", lower: "ArrayLike", upper: "ArrayLike", lmass: "ArrayLike", umass: "ArrayLike", backend: "Backend" = None
) -> "Array":
    """Compute the CRPS for the generalised truncated and censored logistic distribution."""
    B = backends.active if backend is None else backends[backend]
    obs, mu, sigma, lower, upper, lmass, umass = map(B.asarray, (obs, location, scale, lower, upper, lmass, umass))
    ω = (obs - mu) / sigma
    u = (upper - mu) / sigma
    l = (lower - mu) / sigma
    z = B.minimum(B.maximum(ω, l), u)
    F_u = _logis_cdf(u, backend=backend)
    F_l = _logis_cdf(l, backend=backend)
    F_mu = _logis_cdf(-u, backend=backend)
    F_ml = _logis_cdf(-l, backend=backend)
    F_mz = _logis_cdf(-z, backend=backend)

    u_inf = u == float("inf")
    l_inf = l == float("-inf")

    F_mu = B.where(u_inf | l_inf, B.nan, F_mu)
    F_ml = B.where(u_inf | l_inf, B.nan, F_ml)
    u = B.where(u_inf, B.nan, u)
    l = B.where(l_inf, B.nan, l)

    G_u = B.where(u_inf, 0.0, u * F_u + B.log(F_mu))
    G_l = B.where(l_inf, 0.0, l * F_l + B.log(F_ml))
    H_u = B.where(u_inf, 1.0, F_u - u * F_u**2 + (1 - 2 * F_u) * B.log(F_mu))
    H_l = B.where(l_inf, 0.0, F_l - l * F_l**2 + (1 - 2 * F_l) * B.log(F_ml))   

    c = (1 - lmass - umass) / (F_u - F_l)

    s1_u = B.where(u_inf and umass == 0.0, 0.0, u * umass**2)
    s1_l = B.where(l_inf and lmass == 0.0, 0.0, l * lmass**2)

    s1 = B.abs(ω - z) + s1_u - s1_l
    s2 = c * z * ((1 - 2 * lmass) * F_u + (1 - 2 * umass) * F_l) / (1 - lmass - umass)
    s3 = c * (2 * B.log(F_mz) - 2 * G_u * umass - 2 * G_l * lmass)
    s4 = c**2 * (H_u - H_l)
    return sigma * (s1 - s2 - s3 - s4)


def gtcnormal(
    obs: "ArrayLike", location: "ArrayLike", scale: "ArrayLike", lower: "ArrayLike", upper: "ArrayLike", lmass: "ArrayLike", umass: "ArrayLike", backend: "Backend" = None
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

    u_inf = u == float("inf")
    l_inf = l == float("-inf")

    u = B.where(u_inf, B.nan, u)
    l = B.where(l_inf, B.nan, l)
    s1_u = B.where(u_inf and umass == 0.0, 0.0, u * umass**2)
    s1_l = B.where(l_inf and lmass == 0.0, 0.0, l * lmass**2)

    c = (1 - lmass - umass) / (F_u - F_l)

    s1 = B.abs(ω - z) + s1_u - s1_l
    s2 = c * z * (2 * F_z - ((1 - 2 * lmass) * F_u + (1 - 2 * umass) * F_l) / (1 - lmass - umass))
    s3 = c * (2 * f_z - 2 * f_u * umass - 2 * f_l * lmass)
    s4 = c**2 * (F_u2 - F_l2) / B.sqrt(B.pi)
    return sigma * (s1 + s2 + s3 - s4)


def hypergeometric(
    obs: "ArrayLike",
    m: "ArrayLike",
    n: "ArrayLike",
    k: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Compute the CRPS for the hypergeometric distribution.

    We take as inputs the arguments as defined in R's scoringRules package:

    obs: The observed values.
    m: number of success states in the population.
    n: number of failure states in the population.
    k: number of draws, without replacemen, from the population.

    But we follow scipy.stats.hypergeom.pmf for pdf and cdf:

    k or obs: number of observed successes.
    M: total population size.
    n: number of success states in the population.
    N: sample size (number of draws).
    """
    B = backends.active if backend is None else backends[backend]
    obs, m, n, k = map(B.asarray, (obs, m, n, k))

    # scipy uses different notation
    M = m + n
    N = k

    # if n is a scalar, x always has the same shape, which simplifies the computation
    if B.size(n) == 1:
        x = B.arange(n + 1)
        out_ndims = B.max(B.asarray([_input.ndim for _input in [obs, M, m, N]]), axis=0)
        x = B.expand_dims(x, axis=tuple(range(-out_ndims, 0)))
        x, M, m, N = B.broadcast_arrays(x, M, m, N)
        f_np = _hypergeo_pdf(x, M, m, N, backend=backend)
        F_np = _hypergeo_cdf(x, M, m, N, backend=backend)
        s = 2 * B.sum(
            f_np * (B.asarray((obs < x), dtype=float) - F_np + f_np / 2) * (x - obs),
            axis=0,
        )
    # if n is an array, we need to loop over the elements
    else:
        obs, M, m, N = B.broadcast_arrays(obs, M, m, N)
        s = []
        for i, _n in enumerate(n.view(-1)):
            x = B.arange(_n + 1)
            f_np = _hypergeo_pdf(
                x, M.view(-1)[i], m.view(-1)[i], N.view(-1)[i], backend=backend
            )
            F_np = _hypergeo_cdf(
                x, M.view(-1)[i], m.view(-1)[i], N.view(-1)[i], backend=backend
            )
            s.append(
                2
                * B.sum(
                    f_np
                    * (B.asarray((obs.view(-1)[i] < x), dtype=float) - F_np + f_np / 2)
                    * (x - obs.view(-1)[i]),
                    axis=0,
                )
            )
        s = B.asarray(s).reshape(obs.shape)
    return s


def laplace(
    obs: "ArrayLike",
    location: "ArrayLike",
    scale: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Compute the CRPS for the laplace distribution."""
    B = backends.active if backend is None else backends[backend]
    obs, mu, sigma = map(B.asarray, (obs, location, scale))
    obs = (obs - mu) / sigma
    return sigma * (B.abs(obs) + B.exp(-B.abs(obs)) - 3 / 4)


def loglaplace(
    obs: "ArrayLike",
    locationlog: "ArrayLike",
    scalelog: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Compute the CRPS for the log-laplace distribution."""
    B = backends.active if backend is None else backends[backend]
    obs, mulog, sigmalog = map(B.asarray, (obs, locationlog, scalelog))
    obs, mulog, sigmalog = B.broadcast_arrays(obs, mulog, sigmalog)

    logx_norm = (B.log(obs) - mulog) / sigmalog

    cond_0 = obs <= 0.0
    cond_1 = obs < B.exp(mulog)

    F_case_0 = B.asarray(cond_0, dtype=int)
    F_case_1 = B.asarray(~cond_0 & cond_1, dtype=int)
    F_case_2 = B.asarray(~cond_1, dtype=int)
    F = (
        F_case_0 * 0.0
        + F_case_1 * (0.5 * B.exp(logx_norm))
        + F_case_2 * (1 - 0.5 * B.exp(-logx_norm))
    )

    A_case_0 = B.asarray(cond_1, dtype=int)
    A_case_1 = B.asarray(~cond_1, dtype=int)
    A = A_case_0 * 1 / (1 + sigmalog) * (
        1 - (2 * F) ** (1 + sigmalog)
    ) + A_case_1 * -1 / (1 - sigmalog) * (1 - (2 * (1 - F)) ** (1 - sigmalog))

    s = obs * (2 * F - 1) + B.exp(mulog) * (A + sigmalog / (4 - sigmalog**2))
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


def loglogistic(
    obs: "ArrayLike",
    mulog: "ArrayLike",
    sigmalog: "ArrayLike",
    backend: "Backend" = None,
) -> "Array":
    """Compute the CRPS for the log-logistic distribution."""
    B = backends.active if backend is None else backends[backend]
    mulog, sigmalog, obs = map(B.asarray, (mulog, sigmalog, obs))
    F_ms = 1 / (1 + B.exp(-(B.log(obs) - mulog) / sigmalog))
    b = B.beta(1 + sigmalog, 1 - sigmalog)
    I_B = B.betainc(1 + sigmalog, 1 - sigmalog, F_ms)
    s = obs * (2 * F_ms - 1) - B.exp(mulog) * b * (2 * I_B + sigmalog - 1)
    return s


def t(
    obs: "ArrayLike", df: "ArrayLike", location: "ArrayLike", scale: "ArrayLike", backend: "Backend" = None
) -> "Array":
    """Compute the CRPS for the t distribution."""
    B = backends.active if backend is None else backends[backend]
    df, mu, sigma, obs = map(B.asarray, (df, location, scale, obs))
    z = (obs - mu) / sigma
    F_z = _t_cdf(z, df, backend=backend)
    f_z = _t_pdf(z, df, backend=backend)
    G_z = (df + z**2) / (df - 1)    
    s1 = z * (2 * F_z - 1)
    s2 = 2 * f_z * G_z
    s3 = (2 * B.sqrt(df) / (df - 1)) * B.beta(1 / 2, df - 1 / 2) / (B.beta(1 / 2, df / 2)**2)
    return sigma * (s1 + s2 - s3)


def uniform(
    obs: "ArrayLike", min: "ArrayLike", max: "ArrayLike", lmass: "ArrayLike", umass: "ArrayLike", backend: "Backend" = None
) -> "Array":
    """Compute the CRPS for the uniform distribution."""
    B = backends.active if backend is None else backends[backend]
    min, max, lmass, umass, obs = map(B.asarray, (min, max, lmass, umass, obs))
    ω = (obs - min) / (max - min)
    F_ω = B.minimum(B.maximum(ω, B.asarray(0)), B.asarray(1))
    s = B.abs(ω - F_ω) + (F_ω**2) * (1 - lmass - umass)  - F_ω * (1 - 2 * lmass) + ((1 - lmass - umass)**2) / 3 + (1 - lmass) * umass
    return (max - min) * s


def _is_scalar_value(x, value):
    if x.size != 1:
        return False
    return x.item() == value
