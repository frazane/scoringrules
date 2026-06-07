import typing as tp

from scoringrules.core.stats_xp import (
    _binom_cdf,
    _binom_pdf,
    _exp_cdf,
    _gamma_cdf,
    _gev_cdf,
    _gpd_cdf,
    _hypergeo_cdf,
    _hypergeo_pdf,
    _logis_cdf,
    _negbinom_cdf,
    _norm_cdf,
    _norm_pdf,
    _pois_cdf,
    _pois_pdf,
    _t_cdf,
    _t_pdf,
)

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike


def beta(
    obs: "ArrayLike",
    a: "ArrayLike",
    b: "ArrayLike",
    lower: "ArrayLike" = 0.0,
    upper: "ArrayLike" = 1.0,
    *,
    xp,
) -> "Array":
    """Compute the CRPS for the beta distribution."""
    obs, a, b, lower, upper = map(xp.asarray, (obs, a, b, lower, upper))

    if _is_scalar_value(lower, 0.0) and _is_scalar_value(upper, 1.0):
        special_limits = False
    else:
        if xp.any(lower >= upper):
            raise ValueError("lower must be less than upper")
        special_limits = True

    if special_limits:
        obs = (obs - lower) / (upper - lower)

    I_ab = xp.betainc(a, b, obs)
    I_a1b = xp.betainc(a + 1, b, obs)
    F_ab = xp.minimum(xp.maximum(I_ab, 0), 1)
    F_a1b = xp.minimum(xp.maximum(I_a1b, 0), 1)
    bet_rat = 2 * xp.beta(2 * a, 2 * b) / (a * xp.beta(a, b) ** 2)
    s = obs * (2 * F_ab - 1) + (a / (a + b)) * (1 - 2 * F_a1b - bet_rat)

    if special_limits:
        s = s * (upper - lower)

    return s


def binomial(
    obs: "ArrayLike",
    n: "ArrayLike",
    prob: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the CRPS for the binomial distribution.

    Note
    ----
    This is a bit of a hacky implementation, due to how the arrays
    must be broadcasted, but it should work for now.
    """
    obs, n, prob = map(xp.asarray, (obs, n, prob))
    ones_like_n = 0.0 * n + 1

    def _inner(params):
        obs, n, prob = params
        x = xp.arange(0, n + 1)
        w = _binom_pdf(x, n, prob, xp=xp)
        a = _binom_cdf(x, n, prob, xp=xp) - 0.5 * w
        s = 2 * xp.sum(w * ((obs < x) - a) * (x - obs))
        return s

    # if n is a scalar, then if needed we must broadcast k and p to the same shape as n
    # TODO: implement xp.broadcast() for backends
    if n.size == 1:
        x = xp.arange(0, n + 1)
        need_broadcast = not (obs.size == 1 and prob.size == 1)

        if need_broadcast:
            obs = obs[:, None] if obs.size > 1 else obs[None]
            prob = prob[:, None] if prob.size > 1 else prob[None]
            x = x[None]
            x = x * ones_like_n
            prob = prob * ones_like_n
            obs = obs * ones_like_n

        w = _binom_pdf(x, n, prob, xp=xp)
        a = _binom_cdf(x, n, prob, xp=xp) - 0.5 * w
        s = 2 * xp.sum(
            w * ((obs < x) - a) * (x - obs), axis=-1 if need_broadcast else None
        )

    # otherwise, since x would have variable sizes, we must apply the function along the axis
    else:
        obs = obs * ones_like_n if obs.size == 1 else obs
        prob = prob * ones_like_n if prob.size == 1 else prob

        # option 1: in a loop
        s = xp.stack(
            [_inner(params) for params in zip(obs, n, prob, strict=True)],
            axis=-1,
        )

        # option 2: apply_along_axis (does not work with JAX)
        # s = xp.apply_along_axis(_inner, xp.stack((obs, n, prob), axis=-1), -1)

    return s


def exponential(obs: "ArrayLike", rate: "ArrayLike", *, xp) -> "Array":
    """Compute the CRPS for the exponential distribution."""
    rate, obs = map(xp.asarray, (rate, obs))
    s = xp.abs(obs) - (2 * _exp_cdf(obs, rate, xp=xp) / rate) + 1 / (2 * rate)
    return s


def exponentialM(
    obs: "ArrayLike",
    mass: "ArrayLike",
    location: "ArrayLike",
    scale: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the CRPS for the standard exponential distribution with a point mass at the boundary."""
    obs, location, scale, mass = map(xp.asarray, (obs, location, scale, mass))

    if not _is_scalar_value(location, 0.0):
        obs -= location

    a = 1.0 if _is_scalar_value(mass, 0.0) else 1 - mass
    s = xp.abs(obs)

    if _is_scalar_value(scale, 1.0):
        s -= a * (2 * _exp_cdf(obs, 1.0, xp=xp) - 0.5 * a)
    else:
        s -= scale * a * (2 * _exp_cdf(obs, 1 / scale, xp=xp) - 0.5 * a)

    return s


def twopexponential(
    obs: "ArrayLike",
    scale1: "ArrayLike",
    scale2: "ArrayLike",
    location: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the CRPS for the two-piece exponential distribution."""
    scale1, scale2, location, obs = map(xp.asarray, (scale1, scale2, location, obs))
    obs = obs - location
    z = xp.abs(obs)
    c1 = 2 * (scale1**2) / (scale1 + scale2)
    c2 = 2 * (scale2**2) / (scale1 + scale2)
    c3 = (scale1**3 + scale2**3) / (2 * (scale1 + scale2) ** 2)
    s_1 = z + c1 * xp.exp(-z / scale1) - c1 + c3
    s_2 = z + c2 * xp.exp(-z / scale2) - c2 + c3
    s = xp.where(obs < 0.0, s_1, s_2)
    return s


def gamma(
    obs: "ArrayLike",
    shape: "ArrayLike",
    rate: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the CRPS for the gamma distribution."""
    obs, shape, rate = map(xp.asarray, (obs, shape, rate))
    F_ab = _gamma_cdf(obs, shape, rate, xp=xp)
    F_ab1 = _gamma_cdf(obs, shape + 1, rate, xp=xp)
    s = (
        obs * (2 * F_ab - 1)
        - (shape / rate) * (2 * F_ab1 - 1)
        - 1 / (rate * xp.beta(xp.asarray(0.5), shape))
    )
    return s


def csg0(
    obs: "ArrayLike",
    shape: "ArrayLike",
    rate: "ArrayLike",
    shift: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the CRPS for the censored, shifted gamma distribution."""
    obs, shape, rate, shift = map(xp.asarray, (obs, shape, rate, shift))
    obs_shifted = obs + shift
    F_ab_shifted = _gamma_cdf(obs_shifted, shape, rate, xp=xp)
    F_2ab_2d = _gamma_cdf(2 * shift, 2 * shape, rate, xp=xp)
    F_ab_d = _gamma_cdf(shift, shape, rate, xp=xp)
    F_ab1_d = _gamma_cdf(shift, shape + 1, rate, xp=xp)
    F_ab1_shifted = _gamma_cdf(obs_shifted, shape + 1, rate, xp=xp)
    s = (
        obs_shifted * (2 * F_ab_shifted - 1)
        - (shape / (rate * xp.pi))
        * xp.beta(xp.asarray(0.5), shape + 0.5)
        * (1 - F_2ab_2d)
        + shape / rate * (1 + 2 * F_ab_d * F_ab1_d - F_ab_d**2 - 2 * F_ab1_shifted)
        - shift * F_ab_d**2
    )
    return s


EULERMASCHERONI = 0.57721566490153286060651209008240243


def gev(
    obs: "ArrayLike",
    shape: "ArrayLike",
    location: "ArrayLike",
    scale: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the CRPS for the GEV distribution."""
    obs, shape, location, scale = map(xp.asarray, (obs, shape, location, scale))

    obs = (obs - location) / scale
    # if not _is_scalar_value(location, 0.0):
    # obs -= location

    # if not _is_scalar_value(scale, 1.0):
    # obs /= scale

    def _gev_adjust_fn(s, xi, f_xi):
        res = xp.nan * s
        p_xi = xi > 0
        n_xi = xi < 0
        n_inv_xi = -1 / xi

        gen_res = n_inv_xi * f_xi + xp.gammauinc(1 - xi, -xp.log(f_xi)) / xi

        res = xp.where(p_xi & (s <= n_inv_xi), 0, res)
        res = xp.where(p_xi & (s > n_inv_xi), gen_res, res)

        res = xp.where(n_xi & (s < n_inv_xi), gen_res, res)
        res = xp.where(n_xi & (s >= n_inv_xi), n_inv_xi + xp.gamma(1 - xi) / xi, res)

        return res

    F_xi = _gev_cdf(obs, shape, xp=xp)
    zero_shape = shape == 0.0
    shape = xp.where(~zero_shape, shape, xp.nan)
    G_xi = _gev_adjust_fn(obs, shape, F_xi)

    out = xp.where(
        zero_shape,
        -obs - 2 * xp.expi(xp.log(F_xi)) + EULERMASCHERONI - xp.log(2),
        obs * (2 * F_xi - 1)
        - 2 * G_xi
        - (1 - (2 - 2**shape) * xp.gamma(1 - shape)) / shape,
    )

    out = out * scale

    return float(out) if out.size == 1 else out


def gpd(
    obs: "ArrayLike",
    shape: "ArrayLike",
    location: "ArrayLike",
    scale: "ArrayLike",
    mass: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the CRPS for the GPD distribution."""
    shape, location, scale, mass, obs = map(
        xp.asarray, (shape, location, scale, mass, obs)
    )
    shape = xp.where(shape < 1.0, shape, xp.nan)
    mass = xp.where((mass >= 0.0) & (mass <= 1.0), mass, xp.nan)
    ω = (obs - location) / scale
    F_xi = _gpd_cdf(ω, shape, xp=xp)
    s = (
        xp.abs(ω)
        - 2 * (1 - mass) * (1 - (1 - F_xi) ** (1 - shape)) / (1 - shape)
        + ((1 - mass) ** 2) / (2 - shape)
    )
    return scale * s


def gtclogistic(
    obs: "ArrayLike",
    location: "ArrayLike",
    scale: "ArrayLike",
    lower: "ArrayLike",
    upper: "ArrayLike",
    lmass: "ArrayLike",
    umass: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the CRPS for the generalised truncated and censored logistic distribution."""
    obs, mu, sigma, lower, upper, lmass, umass = map(
        xp.asarray, (obs, location, scale, lower, upper, lmass, umass)
    )
    ω = (obs - mu) / sigma
    u = (upper - mu) / sigma
    l = (lower - mu) / sigma
    z = xp.minimum(xp.maximum(ω, l), u)
    F_u = _logis_cdf(u, xp=xp)
    F_l = _logis_cdf(l, xp=xp)
    F_mu = _logis_cdf(-u, xp=xp)
    F_ml = _logis_cdf(-l, xp=xp)
    F_mz = _logis_cdf(-z, xp=xp)

    u_inf = u == float("inf")
    l_inf = l == float("-inf")

    F_mu = xp.where(u_inf | l_inf, xp.nan, F_mu)
    F_ml = xp.where(u_inf | l_inf, xp.nan, F_ml)
    u = xp.where(u_inf, xp.nan, u)
    l = xp.where(l_inf, xp.nan, l)

    G_u = xp.where(u_inf, 0.0, u * F_u + xp.log(F_mu))
    G_l = xp.where(l_inf, 0.0, l * F_l + xp.log(F_ml))
    H_u = xp.where(u_inf, 1.0, F_u - u * F_u**2 + (1 - 2 * F_u) * xp.log(F_mu))
    H_l = xp.where(l_inf, 0.0, F_l - l * F_l**2 + (1 - 2 * F_l) * xp.log(F_ml))

    c = (1 - lmass - umass) / (F_u - F_l)

    s1_u = xp.where(u_inf & (umass == 0.0), 0.0, u * umass**2)
    s1_l = xp.where(l_inf & (lmass == 0.0), 0.0, l * lmass**2)

    s1 = xp.abs(ω - z) + s1_u - s1_l
    s2 = c * z * ((1 - 2 * lmass) * F_u + (1 - 2 * umass) * F_l) / (1 - lmass - umass)
    s3 = c * (2 * xp.log(F_mz) - 2 * G_u * umass - 2 * G_l * lmass)
    s4 = c**2 * (H_u - H_l)
    return sigma * (s1 - s2 - s3 - s4)


def gtcnormal(
    obs: "ArrayLike",
    location: "ArrayLike",
    scale: "ArrayLike",
    lower: "ArrayLike",
    upper: "ArrayLike",
    lmass: "ArrayLike",
    umass: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the CRPS for the generalised truncated and censored normal distribution."""
    mu, sigma, lower, upper, lmass, umass, obs = map(
        xp.asarray, (location, scale, lower, upper, lmass, umass, obs)
    )
    ω = (obs - mu) / sigma
    u = (upper - mu) / sigma
    l = (lower - mu) / sigma
    z = xp.minimum(xp.maximum(ω, l), u)
    F_u = _norm_cdf(u, xp=xp)
    F_l = _norm_cdf(l, xp=xp)
    F_z = _norm_cdf(z, xp=xp)
    F_u2 = _norm_cdf(u * xp.sqrt(2), xp=xp)
    F_l2 = _norm_cdf(l * xp.sqrt(2), xp=xp)
    f_u = _norm_pdf(u, xp=xp)
    f_l = _norm_pdf(l, xp=xp)
    f_z = _norm_pdf(z, xp=xp)

    u_inf = u == float("inf")
    l_inf = l == float("-inf")

    u = xp.where(u_inf, xp.nan, u)
    l = xp.where(l_inf, xp.nan, l)
    s1_u = xp.where(u_inf & (umass == 0.0), 0.0, u * umass**2)
    s1_l = xp.where(l_inf & (lmass == 0.0), 0.0, l * lmass**2)

    c = (1 - lmass - umass) / (F_u - F_l)

    s1 = xp.abs(ω - z) + s1_u - s1_l
    s2 = (
        c
        * z
        * (
            2 * F_z
            - ((1 - 2 * lmass) * F_u + (1 - 2 * umass) * F_l) / (1 - lmass - umass)
        )
    )
    s3 = c * (2 * f_z - 2 * f_u * umass - 2 * f_l * lmass)
    s4 = c**2 * (F_u2 - F_l2) / xp.sqrt(xp.pi)
    return sigma * (s1 + s2 + s3 - s4)


def gtct(
    obs: "ArrayLike",
    df: "ArrayLike",
    location: "ArrayLike",
    scale: "ArrayLike",
    lower: "ArrayLike",
    upper: "ArrayLike",
    lmass: "ArrayLike",
    umass: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the CRPS for the generalised truncated and censored t distribution."""
    df, mu, sigma, lower, upper, lmass, umass, obs = map(
        xp.asarray, (df, location, scale, lower, upper, lmass, umass, obs)
    )
    ω = (obs - mu) / sigma
    u = (upper - mu) / sigma
    l = (lower - mu) / sigma
    z = xp.minimum(xp.maximum(ω, l), u)
    F_u = _t_cdf(u, df, xp=xp)
    F_l = _t_cdf(l, df, xp=xp)
    F_z = _t_cdf(z, df, xp=xp)
    f_u = _t_pdf(u, df, xp=xp)
    f_l = _t_pdf(l, df, xp=xp)
    f_z = _t_pdf(z, df, xp=xp)

    u_inf = u == float("inf")
    l_inf = l == float("-inf")
    u = xp.where(u_inf, xp.nan, u)
    l = xp.where(l_inf, xp.nan, l)

    s1_u = xp.where(u_inf & (umass == 0.0), 0.0, u * umass**2)
    s1_l = xp.where(l_inf & (lmass == 0.0), 0.0, l * lmass**2)

    G_u = xp.where(u_inf, 0.0, -f_u * (df + u**2) / (df - 1))
    G_l = xp.where(l_inf, 0.0, -f_l * (df + l**2) / (df - 1))
    G_z = -f_z * (df + z**2) / (df - 1)

    I_u = xp.where(u_inf, 1.0, xp.betainc(1 / 2, df - 1 / 2, (u**2) / (df + u**2)))
    I_l = xp.where(l_inf, 1.0, xp.betainc(1 / 2, df - 1 / 2, (l**2) / (df + l**2)))
    sgn_u = xp.where(u_inf, 1.0, (u / xp.abs(u)))
    sgn_l = xp.where(l_inf, -1.0, (l / xp.abs(l)))
    H_u = (sgn_u * I_u + 1) / 2
    H_l = (sgn_l * I_l + 1) / 2

    Bbar = (
        (2 * xp.sqrt(df) / (df - 1))
        * xp.beta(1 / 2, df - 1 / 2)
        / (xp.beta(1 / 2, df / 2) ** 2)
    )

    c = (1 - lmass - umass) / (F_u - F_l)

    s1 = xp.abs(ω - z) + s1_u - s1_l
    s2 = (
        c
        * z
        * (
            2 * F_z
            - ((1 - 2 * lmass) * F_u + (1 - 2 * umass) * F_l) / (1 - lmass - umass)
        )
    )
    s3 = 2 * c * (G_z - G_u * umass - G_l * lmass)
    s4 = c**2 * Bbar * (H_u - H_l)
    return sigma * (s1 + s2 - s3 - s4)


def hypergeometric(
    obs: "ArrayLike",
    m: "ArrayLike",
    n: "ArrayLike",
    k: "ArrayLike",
    *,
    xp,
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
    obs, m, n, k = map(xp.asarray, (obs, m, n, k))

    # scipy uses different notation
    M = m + n
    N = k

    # if n is a scalar, x always has the same shape, which simplifies the computation
    if xp.size(n) == 1:
        x = xp.arange(0, n + 1)
        out_ndims = xp.max(
            xp.asarray([_input.ndim for _input in [obs, M, m, N]]), axis=0
        )
        x = xp.expand_dims(x, axis=tuple(range(-out_ndims, 0)))
        x, M, m, N = xp.broadcast_arrays(x, M, m, N)
        f_np = _hypergeo_pdf(x, M, m, N, xp=xp)
        F_np = _hypergeo_cdf(x, M, m, N, xp=xp)
        s = 2 * xp.sum(
            f_np * (xp.asarray((obs < x), dtype=float) - F_np + f_np / 2) * (x - obs),
            axis=0,
        )
    # if n is an array, we need to loop over the elements
    else:
        obs, M, m, N = xp.broadcast_arrays(obs, M, m, N)
        s = []
        for i, _n in enumerate(n.reshape(-1)):
            x = xp.arange(_n + 1)
            f_np = _hypergeo_pdf(
                x, M.reshape(-1)[i], m.reshape(-1)[i], N.reshape(-1)[i], xp=xp
            )
            F_np = _hypergeo_cdf(
                x, M.reshape(-1)[i], m.reshape(-1)[i], N.reshape(-1)[i], xp=xp
            )
            s.append(
                2
                * xp.sum(
                    f_np
                    * (
                        xp.asarray((obs.reshape(-1)[i] < x), dtype=float)
                        - F_np
                        + f_np / 2
                    )
                    * (x - obs.reshape(-1)[i]),
                    axis=0,
                )
            )
        s = xp.asarray(s).reshape(obs.shape)
    return s


def laplace(
    obs: "ArrayLike",
    location: "ArrayLike",
    scale: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the CRPS for the laplace distribution."""
    obs, mu, sigma = map(xp.asarray, (obs, location, scale))
    obs = (obs - mu) / sigma
    return sigma * (xp.abs(obs) + xp.exp(-xp.abs(obs)) - 3 / 4)


def logistic(
    obs: "ArrayLike",
    mu: "ArrayLike",
    sigma: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the CRPS for the normal distribution."""
    mu, sigma, obs = map(xp.asarray, (mu, sigma, obs))
    ω = (obs - mu) / sigma
    return sigma * (ω - 2 * xp.log(_logis_cdf(ω, xp=xp)) - 1)


def loglaplace(
    obs: "ArrayLike",
    locationlog: "ArrayLike",
    scalelog: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the CRPS for the log-laplace distribution."""
    obs, mulog, sigmalog = map(xp.asarray, (obs, locationlog, scalelog))
    obs, mulog, sigmalog = xp.broadcast_arrays(obs, mulog, sigmalog)

    logx_norm = (xp.log(obs) - mulog) / sigmalog

    cond_0 = obs <= 0.0
    cond_1 = obs < xp.exp(mulog)

    F_case_0 = xp.asarray(cond_0, dtype=int)
    F_case_1 = xp.asarray(~cond_0 & cond_1, dtype=int)
    F_case_2 = xp.asarray(~cond_1, dtype=int)
    F = (
        F_case_0 * 0.0
        + F_case_1 * (0.5 * xp.exp(logx_norm))
        + F_case_2 * (1 - 0.5 * xp.exp(-logx_norm))
    )

    A_case_0 = xp.asarray(cond_1, dtype=int)
    A_case_1 = xp.asarray(~cond_1, dtype=int)
    A = A_case_0 * 1 / (1 + sigmalog) * (
        1 - (2 * F) ** (1 + sigmalog)
    ) + A_case_1 * -1 / (1 - sigmalog) * (1 - (2 * (1 - F)) ** (1 - sigmalog))

    s = obs * (2 * F - 1) + xp.exp(mulog) * (A + sigmalog / (4 - sigmalog**2))
    return s


def loglogistic(
    obs: "ArrayLike",
    mulog: "ArrayLike",
    sigmalog: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the CRPS for the log-logistic distribution."""
    mulog, sigmalog, obs = map(xp.asarray, (mulog, sigmalog, obs))
    F_ms = 1 / (1 + xp.exp(-(xp.log(obs) - mulog) / sigmalog))
    b = xp.beta(1 + sigmalog, 1 - sigmalog)
    I_B = xp.betainc(1 + sigmalog, 1 - sigmalog, F_ms)
    s = obs * (2 * F_ms - 1) - xp.exp(mulog) * b * (2 * I_B + sigmalog - 1)
    return s


def lognormal(
    obs: "ArrayLike",
    mulog: "ArrayLike",
    sigmalog: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the CRPS for the lognormal distribution."""
    mulog, sigmalog, obs = map(xp.asarray, (mulog, sigmalog, obs))
    ω = (xp.log(obs) - mulog) / sigmalog
    ex = 2 * xp.exp(mulog + sigmalog**2 / 2)
    return obs * (2.0 * _norm_cdf(ω, xp=xp) - 1) - ex * (
        _norm_cdf(ω - sigmalog, xp=xp)
        + _norm_cdf(sigmalog / xp.sqrt(xp.asarray(2.0)), xp=xp)
        - 1
    )


def mixnorm(
    obs: "ArrayLike",
    m: "ArrayLike",
    s: "ArrayLike",
    w: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the CRPS for a mixture of normal distributions."""
    m, s, w, obs = map(xp.asarray, (m, s, w, obs))

    m_y = obs[..., None] - m
    m_X = m[..., None] - m[..., None, :]
    s_X = xp.sqrt(s[..., None] ** 2 + s[..., None, :] ** 2)
    w_X = w[..., None] * w[..., None, :]

    A_y = m_y * (2 * _norm_cdf(m_y / s, xp=xp) - 1) + 2 * s * _norm_pdf(m_y / s, xp=xp)
    A_X = m_X * (2 * _norm_cdf(m_X / s_X, xp=xp) - 1) + 2 * s_X * _norm_pdf(
        m_X / s_X, xp=xp
    )

    sc_1 = xp.sum(w * A_y, axis=-1)
    sc_2 = xp.sum(w_X * A_X, axis=(-1, -2))

    return sc_1 - 0.5 * sc_2


def negbinom(
    obs: "ArrayLike",
    n: "ArrayLike",
    prob: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the CRPS for the negative binomial distribution."""
    n, prob, obs = map(xp.asarray, (n, prob, obs))
    F_np = _negbinom_cdf(obs, n, prob, xp=xp)
    F_n1p = _negbinom_cdf(obs - 1, n + 1, prob, xp=xp)
    F2 = xp.hypergeometric(n + 1, 1 / 2, 2, -4 * (1 - prob) / (prob**2))
    s = obs * (2 * F_np - 1) - n * (1 - prob) * (prob * (2 * F_n1p - 1) + F2) / (
        prob**2
    )
    return s


def normal(
    obs: "ArrayLike",
    mu: "ArrayLike",
    sigma: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the CRPS for the logistic distribution."""
    mu, sigma, obs = map(xp.asarray, (mu, sigma, obs))
    ω = (obs - mu) / sigma
    return sigma * (
        ω * (2.0 * _norm_cdf(ω, xp=xp) - 1.0)
        + 2.0 * _norm_pdf(ω, xp=xp)
        - 1.0 / xp.sqrt(xp.pi)
    )


def poisson(
    obs: "ArrayLike",
    mean: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the CRPS for the poisson distribution."""
    mean, obs = map(xp.asarray, (mean, obs))
    F_m = _pois_cdf(obs, mean, xp=xp)
    f_m = _pois_pdf(xp.floor(obs), mean, xp=xp)
    I0 = xp.mbessel0(2 * mean)
    I1 = xp.mbessel1(2 * mean)
    s = (
        (obs - mean) * (2 * F_m - 1)
        + 2 * mean * f_m
        - mean * xp.exp(-2 * mean) * (I0 + I1)
    )
    return s


def t(
    obs: "ArrayLike",
    df: "ArrayLike",
    location: "ArrayLike",
    scale: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the CRPS for the t distribution."""
    df, mu, sigma, obs = map(xp.asarray, (df, location, scale, obs))
    z = (obs - mu) / sigma
    F_z = _t_cdf(z, df, xp=xp)
    f_z = _t_pdf(z, df, xp=xp)
    G_z = (df + z**2) / (df - 1)
    s1 = z * (2 * F_z - 1)
    s2 = 2 * f_z * G_z
    s3 = (
        (2 * xp.sqrt(df) / (df - 1))
        * xp.beta(1 / 2, df - 1 / 2)
        / (xp.beta(1 / 2, df / 2) ** 2)
    )
    return sigma * (s1 + s2 - s3)


def uniform(
    obs: "ArrayLike",
    min: "ArrayLike",
    max: "ArrayLike",
    lmass: "ArrayLike",
    umass: "ArrayLike",
    *,
    xp,
) -> "Array":
    """Compute the CRPS for the uniform distribution."""
    min, max, lmass, umass, obs = map(xp.asarray, (min, max, lmass, umass, obs))
    ω = (obs - min) / (max - min)
    F_ω = xp.minimum(xp.maximum(ω, xp.asarray(0)), xp.asarray(1))
    s = (
        xp.abs(ω - F_ω)
        + (F_ω**2) * (1 - lmass - umass)
        - F_ω * (1 - 2 * lmass)
        + ((1 - lmass - umass) ** 2) / 3
        + (1 - lmass) * umass
    )
    return (max - min) * s


def _is_scalar_value(x, value):
    if x.size != 1:
        return False
    return x.item() == value
