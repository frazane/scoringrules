import math

import numpy as np
from numba import guvectorize, njit, vectorize

INV_SQRT_PI = 1 / np.sqrt(np.pi)
EPSILON = 1e-6
EULERMASCHERONI = np.euler_gamma


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:], float64[:])",
    ],
    "(),(a),(a)->()",
)
def quantile_pinball_gufunc(
    obs: np.ndarray, fct: np.ndarray, alpha: np.ndarray, out: np.ndarray
):
    """Pinball loss based estimator for the CRPS from quantiles."""
    obs = obs[0]
    Q = fct.shape[0]
    pb = 0
    for fc, level in zip(fct, alpha):  # noqa: B905
        pb += (obs > fc) * level * (obs - fc) + (obs <= fc) * (1 - level) * (fc - obs)
    out[0] = 2 * (pb / Q)


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:])",
    ],
    "(),(n)->()",
)
def _crps_ensemble_int_gufunc(obs: np.ndarray, fct: np.ndarray, out: np.ndarray):
    """CRPS estimator based on the integral form."""
    obs = obs[0]
    M = fct.shape[0]

    if np.isnan(obs):
        out[0] = np.nan
        return

    obs_cdf = 0
    forecast_cdf = 0.0
    prev_forecast = 0.0
    integral = 0.0

    for n, forecast in enumerate(fct):
        if np.isnan(forecast):
            if n == 0:
                integral = np.nan
            forecast = prev_forecast  # noqa: PLW2901
            break

        if obs_cdf == 0 and obs < forecast:
            # this correctly handles the transition point of the obs CDF
            integral += (obs - prev_forecast) * forecast_cdf**2
            integral += (forecast - obs) * (forecast_cdf - 1) ** 2
            obs_cdf = 1
        else:
            integral += (forecast_cdf - obs_cdf) ** 2 * (forecast - prev_forecast)

        forecast_cdf += 1 / M
        prev_forecast = forecast

    if obs_cdf == 0:
        integral += obs - forecast

    out[0] = integral


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:])",
    ],
    "(),(n)->()",
)
def _crps_ensemble_qd_gufunc(obs: np.ndarray, fct: np.ndarray, out: np.ndarray):
    """CRPS estimator based on the quantile decomposition form."""
    obs = obs[0]
    M = fct.shape[-1]

    if np.isnan(obs):
        out[0] = np.nan
        return

    obs_cdf = 0.0
    integral = 0.0

    for i, forecast in enumerate(fct):
        if obs < forecast:
            obs_cdf = 1.0

        integral += (forecast - obs) * (M * obs_cdf - (i + 1) + 0.5)

    out[0] = (2 / M**2) * integral


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:])",
    ],
    "(),(n)->()",
)
def _crps_ensemble_nrg_gufunc(obs: np.ndarray, fct: np.ndarray, out: np.ndarray):
    """CRPS estimator based on the energy form."""
    obs = obs[0]
    M = fct.shape[-1]

    if np.isnan(obs):
        out[0] = np.nan
        return

    e_1 = 0
    e_2 = 0

    for x_i in fct:
        e_1 += abs(x_i - obs)
        for x_j in fct:
            e_2 += abs(x_i - x_j)

    out[0] = e_1 / M - 0.5 * e_2 / (M**2)


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:])",
    ],
    "(),(n)->()",
)
def _crps_ensemble_fair_gufunc(obs: np.ndarray, fct: np.ndarray, out: np.ndarray):
    """Fair version of the CRPS estimator based on the energy form."""
    obs = obs[0]
    M = fct.shape[-1]

    if np.isnan(obs):
        out[0] = np.nan
        return

    e_1 = 0
    e_2 = 0

    for x_i in fct:
        e_1 += abs(x_i - obs)
        for x_j in fct:
            e_2 += abs(x_i - x_j)

    out[0] = e_1 / M - 0.5 * e_2 / (M * (M - 1))


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:])",
    ],
    "(),(n)->()",
)
def _crps_ensemble_pwm_gufunc(obs: np.ndarray, fct: np.ndarray, out: np.ndarray):
    """CRPS estimator based on the probability weighted moment (PWM) form."""
    obs = obs[0]
    M = fct.shape[-1]

    if np.isnan(obs):
        out[0] = np.nan
        return

    expected_diff = 0.0
    β_0 = 0.0
    β_1 = 0.0

    for i, forecast in enumerate(fct):
        expected_diff += np.abs(forecast - obs)
        β_0 += forecast
        β_1 += forecast * i

    out[0] = expected_diff / M + β_0 / M - 2 * β_1 / (M * (M - 1))


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:])",
    ],
    "(),(n)->()",
)
def _crps_ensemble_akr_gufunc(obs: np.ndarray, fct: np.ndarray, out: np.ndarray):
    """CRPS estimaton based on the approximate kernel representation."""
    M = fct.shape[-1]
    obs = obs[0]
    e_1 = 0
    e_2 = 0
    for i, forecast in enumerate(fct):
        if i == 0:
            i = M - 1
        e_1 += abs(forecast - obs)
        e_2 += abs(forecast - fct[i - 1])
    out[0] = e_1 / M - 0.5 * 1 / M * e_2


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:])",
    ],
    "(),(n)->()",
)
def _crps_ensemble_akr_circperm_gufunc(
    obs: np.ndarray, fct: np.ndarray, out: np.ndarray
):
    """CRPS estimaton based on the AKR with cyclic permutation."""
    M = fct.shape[-1]
    obs = obs[0]
    e_1 = 0.0
    e_2 = 0.0
    for i, forecast in enumerate(fct):
        sigma_i = int((i + 1 + ((M - 1) / 2)) % M)
        e_1 += abs(forecast - obs)
        e_2 += abs(forecast - fct[sigma_i])
    out[0] = e_1 / M - 0.5 * 1 / M * e_2


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:], float64[:], float64[:])",
    ],
    "(),(n),(),(n)->()",
)
def _owcrps_ensemble_nrg_gufunc(
    obs: np.ndarray,
    fct: np.ndarray,
    ow: np.ndarray,
    fw: np.ndarray,
    out: np.ndarray,
):
    """Outcome-weighted CRPS estimator based on the energy form."""
    obs = obs[0]
    ow = ow[0]
    M = fct.shape[-1]

    if np.isnan(obs):
        out[0] = np.nan
        return

    e_1 = 0.0
    e_2 = 0.0

    for i, x_i in enumerate(fct):
        e_1 += abs(x_i - obs) * fw[i] * ow
        for j, x_j in enumerate(fct):
            e_2 += abs(x_i - x_j) * fw[i] * fw[j] * ow

    wbar = np.mean(fw)

    out[0] = e_1 / (M * wbar) - 0.5 * e_2 / ((M * wbar) ** 2)


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:], float64[:], float64[:])",
    ],
    "(),(n),(),(n)->()",
)
def _vrcrps_ensemble_nrg_gufunc(
    obs: np.ndarray,
    fct: np.ndarray,
    ow: np.ndarray,
    fw: np.ndarray,
    out: np.ndarray,
):
    """Vertically re-scaled CRPS estimator based on the energy form."""
    obs = obs[0]
    ow = ow[0]
    M = fct.shape[-1]

    if np.isnan(obs):
        out[0] = np.nan
        return

    e_1 = 0.0
    e_2 = 0.0

    for i, x_i in enumerate(fct):
        e_1 += abs(x_i - obs) * fw[i] * ow
        for j, x_j in enumerate(fct):
            e_2 += abs(x_i - x_j) * fw[i] * fw[j]

    wbar = np.mean(fw)
    wabs_x = np.mean(np.abs(fct) * fw)
    wabs_y = abs(obs) * ow

    out[0] = e_1 / M - 0.5 * e_2 / (M**2) + (wabs_x - wabs_y) * (wbar - ow)


@njit(["float32(float32)", "float64(float64)"])
def _norm_cdf(x: float) -> float:
    """Cumulative distribution function for the standard normal distribution."""
    out: float = (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    return out


@njit(["float32(float32)", "float64(float64)"])
def _norm_pdf(x: float) -> float:
    """Probability density function for the standard normal distribution."""
    out: float = (1 / math.sqrt(2 * math.pi)) * math.exp(-(x**2) / 2)
    return out


@njit(["float32(float32)", "float64(float64)"])
def _logis_cdf(x: float) -> float:
    """Cumulative distribution function for the standard logistic distribution."""
    out: float = 1.0 / (1.0 + math.exp(-x))
    return out


@njit(["float32(float32, float32)", "float64(float64, float64)"])
def _exp_cdf(x: float, rate: float) -> float:
    """Cumulative distribution function for the exponential distribution."""
    out: float = np.maximum(1 - math.exp(- rate * x), 0)
    return out


@njit(["float32(float32, float32, float32)", "float64(float64, float64, float64)"])
def _gamma_cdf(x: float, shape: float, rate: float) -> float:
    """Cumulative distribution function for the gamma distribution."""
    raise NotImplementedError("The gamma function is currently not available for backend numba")
    out: float = np.maximum(np.li_gamma(shape, rate * x) / np.gamma(shape), 0)
    return out


@njit(["float32(float32, float32)", "float64(float64, float64)"])
def _pois_cdf(x: float, mean: float) -> float:
    """Cumulative distribution function for the Poisson distribution."""
    raise NotImplementedError("The gamma function is currently not available for backend numba")
    out: float = np.maximum(np.ui_gamma(np.floor(x + 1), mean) / np.gamma(np.floor(x + 1)), 0)
    return out


@njit(["float32(float32, float32)", "float64(float64, float64)"])
def _pois_pdf(x: float, mean: float) -> float:
    """Probability mass function for the Poisson distribution."""
    raise NotImplementedError("The factorial function is currently not available for backend numba")
    out: float = np.isinteger(x) * (mean**x * np.exp(-x) / np.factorial(x))
    return out

@njit(["float32(float32, float32)", "float64(float64, float64)"])
def _t_cdf(x: float, df: float) -> float:
    """Cumulative distribution function for the standard Student's t distribution."""
    raise NotImplementedError("The hypergeometric function is currently not available for backend numba")
    out: float = 1 / 2 + x * hypergeo(1 / 2, (df + 1) / 2, 3 / 2, - (x**2) / df) / (np.sqrt(df) * beta(1 / 2, df / 2))
    return out

@njit(["float32(float32, float32)", "float64(float64, float64)"])
def _t_pdf(x: float, df: float) -> float:
    """Probability density function for the standard Student's t distribution."""
    raise NotImplementedError("The beta function is currently not available for backend numba")
    out: float = ((1 + x**2 / df)**(- (df + 1) / 2)) / (np.sqrt(df) * beta(1 / 2, df / 2))
    return out

@njit(["float32(float32, float32)", "float64(float64, float64)"])
def _gev_cdf(x: float, shape: float) -> float:
    """Cumulative distribution function for the standard GEV distribution."""
    if shape == 0.0:
        out: float = math.exp(- math.exp(-x))
    elif shape > 0.0:
        if x > (- 1 / shape):
            out: float = math.exp(-(1 + shape * x)**(-1 / shape))
        else:
            out: float = 0.0
    else:
        if x < (- 1 / shape):
            out: float = math.exp(-(1 + shape * x)**(-1 / shape))
        else:
            out: float = 1.0
    return out

@njit(["float32(float32, float32)", "float64(float64, float64)"])
def _gpd_cdf(x: float, shape: float) -> float:
    """Cumulative distribution function for the standard GPD distribution."""
    if shape == 0.0:
        if x < 0:
            out: float = 0.0
        else:
            out: float = 1 - math.exp(-x)
    elif shape > 0.0:
        if x < 0:
            out: float = 0.0
        else:
            out: float = (1 - (1 + shape * x)**(- 1 / shape))
    else:
        if x < 0:
            out: float = 0.0
        elif x < 1 / np.abs(shape):
            out: float = (1 - (1 + shape * x)**(- 1 / shape))
        else:
            out: float = 1.0
    return out


@njit(["float32(float32, int32, float32)", "float64(float64, int64, float64)"])
def _binom_pdf(x: float, n: int, prob: float) -> float:
    """Probability density function for the binomial distribution."""
    raise NotImplementedError("The comb function is currently not available for backend numba")
    out: float = np.isinteger(x) * comb(n, x) * prob**x * (1 - prob)**x
    return out


@njit(["float32(float32, int32, float32)", "float64(float64, int64, float64)"])
def _binom_cdf(x: float, n: int, prob: float) -> float:
    """Cumulative distribution function for the binomial distribution."""
    raise NotImplementedError("The incomplete beta function is currently not available for backend numba")
    out: float = betainc(n - np.floor(x), np.floor(x) + 1, 1 - prob)
    return out


@njit(["float32(float32, int32, int32, int32)", "float64(float64, int64, int64, int64)"])
def _hypergeo_pdf(x: int, m: int, n: int, k: int) -> float:
    """Probability density function for the hypergeometric distribution."""
    raise NotImplementedError("The comb function is currently not available for backend numba")
    out: float = comb(m, x) * comb(n, k - x) / comb(m + n, k)
    return out


@njit(["float32(float32, int32, int32, int32)", "float64(float64, int64, int64, int64)"])
def _hypergeo_cdf(x: float, m: int, n: int, k: int) -> float:
    """Cumulative distribution function for the hypergeometric distribution."""
    raise NotImplementedError("The comb function is currently not available for backend numba")
    out: float = _hypergeo_pdf(0, m, n, k)
    for i in range(np.floor(x)):
        out += _hypergeo_pdf(i, m, n, k)
    return out


@njit(["float32(float32, float32, float32)", "float64(float64, float64, float64)"])
def _negbinom_cdf(x: float, n: float, prob: float) -> float:
    """Cumulative distribution function for the negative binomial distribution."""
    raise NotImplementedError("The incomplete beta function is currently not available for backend numba")
    if x < 0.0:
        out: float = 0.0
    else:
        out: float = betainc(n, np.floor(x + 1), prob)
    return out


@vectorize(["float32(float32, float32)", "float64(float64, float64)"])
def _crps_exponential_ufunc(rate: float, observation: float) -> float:
    out: float = np.abs(observation) - (2 * _exp_cdf(observation, rate) / rate) + 1 / (2 * rate)
    return out


@vectorize(["float32(float32, float32, float32, float32)", "float64(float64, float64, float64, float64)"])
def _crps_exponentialM_ufunc(mass: float, location: float, scale: float, observation: float) -> float:
    ω = (observation - location) / scale
    F_y = np.maximum(1 - math.exp(-ω), 0)
    out: float = np.abs(ω) - 2 * (1 - mass) * F_y + ((1 - mass)**2) / 2
    return out * scale


@vectorize(["float32(float32, float32, float32, float32, float32)", "float64(float64, float64, float64, float64, float64)"])
def _crps_beta_ufunc(shape1: float, shape2: float, lower: float, upper: float, observation: float) -> float:
    raise NotImplementedError("The beta function is currently not available for backend numba")
    obs_std = (observation - lower) / (upper - lower)
    I_ab = np.betainc(shape1, shape2, obs_std)
    I_a1b = np.betainc(shape1 + 1, shape2, obs_std)
    F_ab = np.minimum(np.maximum(I_ab, 0), 1)
    F_a1b = np.minimum(np.maximum(I_a1b, 0), 1)
    bet_rat = 2 * beta(2 * shape1, 2 * shape2) / (shape1 * beta(shape1, shape2)**2)
    out: float = obs_std * (2 * F_ab - 1) + (shape1 / (shape1 + shape2)) * (1 - 2 * F_a1b - bet_rat)
    return out


@vectorize(["float32(float32, float32, float32)", "float64(float64, float64, float64)"])
def _crps_normal_ufunc(obs: float, mu: float, sigma: float) -> float:
    ω = (obs - mu) / sigma
    out: float = sigma * (ω * (2 * _norm_cdf(ω) - 1) + 2 * _norm_pdf(ω) - INV_SQRT_PI)
    return out


@vectorize(["float32(float32, float32, float32)", "float64(float64, float64, float64)"])
def _crps_lognormal_ufunc(obs: float, mulog: float, sigmalog: float) -> float:
    ω = (np.log(obs) - mulog) / sigmalog
    ex = 2 * np.exp(mulog + sigmalog**2 / 2)
    out: float = obs * (2 * _norm_cdf(ω) - 1) - ex * (
        _norm_cdf(ω - sigmalog) + _norm_cdf(sigmalog / np.sqrt(2)) - 1
    )
    return out


@vectorize(["float32(float32, float32, float32)", "float64(float64, float64, float64)"])
def _crps_logistic_ufunc(obs: float, mu: float, sigma: float) -> float:
    ω = (obs - mu) / sigma
    out: float = sigma * (ω - 2 * np.log(_logis_cdf(ω)) - 1)
    return out


estimator_gufuncs = {
    "akr_circperm": _crps_ensemble_akr_circperm_gufunc,
    "akr": _crps_ensemble_akr_gufunc,
    "fair": _crps_ensemble_fair_gufunc,
    "int": _crps_ensemble_int_gufunc,
    "nrg": _crps_ensemble_nrg_gufunc,
    "pwm": _crps_ensemble_pwm_gufunc,
    "qd": _crps_ensemble_qd_gufunc,
    "ownrg": _owcrps_ensemble_nrg_gufunc,
    "vrnrg": _vrcrps_ensemble_nrg_gufunc,
}


__all__ = [
    "_crps_ensemble_akr_circperm_gufunc",
    "_crps_ensemble_akr_gufunc",
    "_crps_ensemble_fair_gufunc",
    "_crps_ensemble_int_gufunc",
    "_crps_ensemble_nrg_gufunc",
    "_crps_ensemble_pwm_gufunc",
    "_crps_ensemble_qd_gufunc",
    "_crps_beta_ufunc",
    "_crps_binomial_ufunc",
    "_crps_exponential_ufunc",
    "_crps_exponentialM_ufunc",
    "_crps_gamma_ufunc",
    "_crps_gev_ufunc",
    "_crps_gpd_ufunc",
    "_crps_gtclogistic_ufunc",
    "_crps_gtcnormal_ufunc",
    "_crps_gtct_ufunc",
    "_crps_hypergeometric_ufunc",
    "_crps_laplace_ufunc",
    "_crps_logistic_ufunc",
    "quantile_pinball_gufunc",
]
