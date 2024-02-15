import math

import numpy as np
from numba import guvectorize, njit, vectorize

INV_SQRT_PI = 1 / np.sqrt(np.pi)
EPSILON = 1e-6


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:])",
    ],
    "(n),()->()",
)
def _crps_ensemble_int_gufunc(
    forecasts: np.ndarray, observation: np.ndarray, out: np.ndarray
):
    """CRPS estimator based on the integral form."""
    obs = observation[0]
    M = forecasts.shape[0]

    if np.isnan(obs):
        out[0] = np.nan
        return

    obs_cdf = 0
    forecast_cdf = 0.0
    prev_forecast = 0.0
    integral = 0.0

    for n, forecast in enumerate(forecasts):
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
    "(n),()->()",
)
def _crps_ensemble_qd_gufunc(
    forecasts: np.ndarray, observation: np.ndarray, out: np.ndarray
):
    """CRPS estimator based on the quantile decomposition form."""
    obs = observation[0]
    M = forecasts.shape[-1]

    if np.isnan(obs):
        out[0] = np.nan
        return

    obs_cdf = 0.0
    integral = 0.0

    for i, forecast in enumerate(forecasts):
        if obs < forecast:
            obs_cdf = 1.0

        integral += (forecast - obs) * (M * obs_cdf - (i + 1) + 0.5)

    out[0] = (2 / M**2) * integral


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:])",
    ],
    "(n),()->()",
)
def _crps_ensemble_nrg_gufunc(
    forecasts: np.ndarray, observation: np.ndarray, out: np.ndarray
):
    """CRPS estimator based on the energy form."""
    obs = observation[0]
    M = forecasts.shape[-1]

    if np.isnan(obs):
        out[0] = np.nan
        return

    e_1 = 0
    e_2 = 0

    for x_i in forecasts:
        e_1 += abs(x_i - obs)
        for x_j in forecasts:
            e_2 += abs(x_i - x_j)

    out[0] = e_1 / M - 0.5 * e_2 / (M**2)


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:])",
    ],
    "(n),()->()",
)
def _crps_ensemble_fair_gufunc(
    forecasts: np.ndarray, observation: np.ndarray, out: np.ndarray
):
    """Fair version of the CRPS estimator based on the energy form."""
    obs = observation[0]
    M = forecasts.shape[-1]

    if np.isnan(obs):
        out[0] = np.nan
        return

    e_1 = 0
    e_2 = 0

    for x_i in forecasts:
        e_1 += abs(x_i - obs)
        for x_j in forecasts:
            e_2 += abs(x_i - x_j)

    out[0] = e_1 / M - 0.5 * e_2 / (M * (M - 1))


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:])",
    ],
    "(n),()->()",
)
def _crps_ensemble_pwm_gufunc(
    forecasts: np.ndarray, observation: np.ndarray, out: np.ndarray
):
    """CRPS estimator based on the probability weighted moment (PWM) form."""
    obs = observation[0]
    M = forecasts.shape[-1]

    if np.isnan(obs):
        out[0] = np.nan
        return

    expected_diff = 0.0
    β_0 = 0.0
    β_1 = 0.0

    for i, forecast in enumerate(forecasts):
        expected_diff += np.abs(forecast - obs)
        β_0 += forecast
        β_1 += forecast * i

    out[0] = expected_diff / M + β_0 / M - 2 * β_1 / (M * (M - 1))


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:])",
    ],
    "(n),()->()",
)
def _crps_ensemble_akr_gufunc(
    forecasts: np.ndarray, observation: np.ndarray, out: np.ndarray
):
    """CRPS estimaton based on the approximate kernel representation."""
    M = forecasts.shape[-1]
    obs = observation[0]
    e_1 = 0
    e_2 = 0
    for i, forecast in enumerate(forecasts):
        if i == 0:
            i = M - 1
        e_1 += abs(forecast - obs)
        e_2 += abs(forecast - forecasts[i - 1])
    out[0] = e_1 / M - 0.5 * 1 / M * e_2


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:])",
    ],
    "(n),()->()",
)
def _crps_ensemble_akr_circperm_gufunc(
    forecasts: np.ndarray, observation: np.ndarray, out: np.ndarray
):
    """CRPS estimaton based on the AKR with cyclic permutation."""
    M = forecasts.shape[-1]
    obs = observation[0]
    e_1 = 0.0
    e_2 = 0.0
    for i, forecast in enumerate(forecasts):
        sigma_i = int((i + 1 + ((M - 1) / 2)) % M)
        e_1 += abs(forecast - obs)
        e_2 += abs(forecast - forecasts[sigma_i])
    out[0] = e_1 / M - 0.5 * 1 / M * e_2


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:], float64[:], float64[:])",
    ],
    "(n),(),(n),()->()",
)
def _owcrps_ensemble_nrg_gufunc(
    forecasts: np.ndarray,
    observation: np.ndarray,
    fw: np.ndarray,
    ow: np.ndarray,
    out: np.ndarray,
):
    """Outcome-weighted CRPS estimator based on the energy form."""
    obs = observation[0]
    ow = ow[0]
    M = forecasts.shape[-1]

    if np.isnan(obs):
        out[0] = np.nan
        return

    e_1 = 0.0
    e_2 = 0.0

    for i, x_i in enumerate(forecasts):
        e_1 += abs(x_i - obs) * fw[i] * ow
        for j, x_j in enumerate(forecasts):
            e_2 += abs(x_i - x_j) * fw[i] * fw[j] * ow

    wbar = np.mean(fw)

    out[0] = e_1 / (M * wbar) - 0.5 * e_2 / ((M * wbar) ** 2)


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:], float64[:], float64[:])",
    ],
    "(n),(),(n),()->()",
)
def _vrcrps_ensemble_nrg_gufunc(
    forecasts: np.ndarray,
    observation: np.ndarray,
    fw: np.ndarray,
    ow: np.ndarray,
    out: np.ndarray,
):
    """Vertically re-scaled CRPS estimator based on the energy form."""
    obs = observation[0]
    ow = ow[0]
    M = forecasts.shape[-1]

    if np.isnan(obs):
        out[0] = np.nan
        return

    e_1 = 0.0
    e_2 = 0.0

    for i, x_i in enumerate(forecasts):
        e_1 += abs(x_i - obs) * fw[i] * ow
        for j, x_j in enumerate(forecasts):
            e_2 += abs(x_i - x_j) * fw[i] * fw[j]

    wbar = np.mean(fw)
    wabs_x = np.mean(np.abs(forecasts) * fw)
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
    out: float = np.maximum(np.li_gamma(shape, rate * x) / np.gamma(shape), 0)
    return out


@njit(["float32(float32, float32)", "float64(float64, float64)"])
def _pois_cdf(x: float, mean: float) -> float:
    """Cumulative distribution function for the Poisson distribution."""
    out: float = np.maximum(np.ui_gamma(np.floor(x + 1), mean) / np.gamma(np.floor(x + 1)), 0)
    return out


@njit(["float32(float32, float32)", "float64(float64, float64)"])
def _pois_pdf(x: float, mean: float) -> float:
    """Probability mass function for the Poisson distribution."""
    out: float = np.isinteger(x) * (mean**x * np.exp(-x) / np.factorial(x))
    return out

@njit(["float32(float32, float32)", "float64(float64, float64)"])
def _t_cdf(x: float, df: float) -> float:
    """Cumulative distribution function for the standard Student's t distribution."""
    out: float = 1 / 2 + x * hypergeo(1 / 2, (df + 1) / 2, 3 / 2, - (x**2) / df) / (np.sqrt(df) * beta(1 / 2, df / 2))
    return out

@njit(["float32(float32, float32)", "float64(float64, float64)"])
def _t_pdf(x: float, df: float) -> float:
    """Probability density function for the standard Student's t distribution."""
    out: float = ((1 + x**2 / df)**(- (df + 1) / 2)) / (np.sqrt(df) * beta(1 / 2, df / 2))
    return out

@vectorize(["float32(float32, float32)", "float64(float64, float64)"])
def _crps_exponential_ufunc(rate: float, observation: float) -> float:
    out: float = np.abs(observation) - (2 * _exp_cdf(observation, rate) / rate) + 1 / (2 * rate)
    return out


@vectorize(["float32(float32, float32, float32, float32, float32)", "float64(float64, float64, float64, float64, float64)"])
def _crps_beta_ufunc(shape1: float, shape2: float, lower: float, upper:float, observation: float) -> float:
    obs_std = (observation - lower) / (upper - lower)
    I_ab = np.betainc(shape1, shape2, obs_std)
    I_a1b = np.betainc(shape1 + 1, shape2, obs_std)
    F_ab = np.minimum(np.maximum(I_ab, 0), 1)
    F_a1b = np.minimum(np.maximum(I_a1b, 0), 1)
    bet_rat = 2 * beta(2 * shape1, 2 * shape2) / (shape1 * beta(shape1, shape2)**2)
    out: float = obs_std * (2 * F_ab - 1) + (shape1 / (shape1 + shape2)) * ( 1 - 2 * F_a1b - bet_rat)
    return out


@vectorize(["float32(float32, float32, float32)", "float64(float64, float64, float64)"])
def _crps_gamma_ufunc(shape: float, rate: float, observation: float) -> float:
    F_ab = _gamma_cdf(observation, shape, rate)
    F_ab1 = _gamma_cdf(observation, shape + 1, rate)
    out: float = observation * (2 * F_ab - 1) + (shape / rate) * (2 * F_ab1 - 1) - 1 / (rate * beta(0.5, shape))
    return out


@vectorize(["float32(float32, float32, float32)", "float64(float64, float64, float64)"])
def _crps_laplace_ufunc(location: float, scale: float, observation: float) -> float:
    ω = (observation - location) / scale
    out: float = scale * (ω + math.exp(-ω) - 3/4)
    return out


@vectorize(["float32(float32, float32, float32)", "float64(float64, float64, float64)"])
def _crps_logistic_ufunc(mu: float, sigma: float, observation: float) -> float:
    ω = (observation - mu) / sigma
    out: float = sigma * (ω - 2 * np.log(_logis_cdf(ω)) - 1)
    return out


@vectorize(["float32(float32, float32, float32)", "float64(float64, float64, float64)"])
def _crps_loglogistic_ufunc(mulog: float, sigmalog: float, observation: float) -> float:
    F_ms = 1 / (1 + np.exp( - (np.log(observation) - mulog) / sigmalog))
    b = beta(1 + sigmalog, 1 - sigmalog)
    I = betainc(1 + sigmalog, 1 - sigmalog, F_ms)
    out: float = observation * (2 * F_ms - 1) - np.exp(mulog) * b * (2*I + sigmalog - 1)
    return out


@vectorize(["float32(float32, float32, float32)", "float64(float64, float64, float64)"])
def _crps_lognormal_ufunc(mulog: float, sigmalog: float, observation: float) -> float:
    ω = (np.log(observation) - mulog) / sigmalog
    ex = 2 * np.exp(mulog + sigmalog**2 / 2)
    out: float = observation * (2 * _norm_cdf(ω) - 1) - ex * (
        _norm_cdf(ω - sigmalog) + _norm_cdf(sigmalog / np.sqrt(2)) - 1
    )
    return out


@vectorize(["float32(float32, float32, float32)", "float64(float64, float64, float64)"])
def _crps_normal_ufunc(mu: float, sigma: float, observation: float) -> float:
    ω = (observation - mu) / sigma
    out: float = sigma * (ω * (2 * _norm_cdf(ω) - 1) + 2 * _norm_pdf(ω) - INV_SQRT_PI)
    return out


@vectorize(["float32(float32, float32)", "float64(float64, float64)"])
def _crps_poisson_ufunc(mean: float, observation: float) -> float:
    F_m = _pois_cdf(observation, mean)
    f_m = _pois_pdf(np.floor(observation), mean)
    I0 = mbessel0(2 * mean)
    I1 = mbessel1(2 * mean)
    out: float = (y - mean) * (2 * F_m - 1) + 2 * mean * f_m - mean * np.exp(-2 * mean) * (I0 + I1)
    return out


@vectorize(["float32(float32, float32, float32, float32, float32)", "float64(float64, float64, float64, float64, float64)"])
def _crps_uniform_ufunc(min: float, max: float, lmass: float, umass: float, observation: float) -> float:
    ω = (observation - min) / (max - min)
    F_ω = np.minimum(np.maximum(ω, 0), 1)
    out: float = np.abs(ω - F_ω) + (F_ω**2) * (1 - lmass - umass)  - F_ω * (1 - 2 * lmass) + ((1 - lmass - umass)**2) / 3 + (1 - lmass) * umass
    return out


@vectorize(["float32(float32, float32, float32, float32)", "float64(float64, float64, float64, float64)"])
def _crps_t_ufunc(df: float, location: float, scale: float, observation: float) -> float:
    ω = (observation - location) / scale
    F_nu = _t_cdf(ω)
    f_nu = _t_pdf(ω)
    out: float = ω * (2 * F_nu - 1) + 2 * f_nu * (df + ω**2) / (df - 1) - (2 * np.sqrt(df) * beta(1 / 2, df - 1)) / ((df - 1) * beta(1 / 2, df / 2)**2)
    return out


@vectorize(["float32(float32, float32, float32, float32)", "float64(float64, float64, float64, float64)"])
def _crps_tpexponential_ufunc(scale1: float, scale2: float, location: float, observation: float) -> float:
    ω = (observation - location)
    scalei = scale2
    if ω < 0: scalei = scale1
    out: float = np.abs(ω) + 2 * (scalei**2) * math.exp( - np.abs(ω) / scalei) / (scale1 + scale2) - 2 * (scalei**2) / (scale1 + scale2) + (scale1**3 + scale2**3) / (2 * (scale1 + scale2)**2)
    return out


@vectorize(["float32(float32, float32, float32, float32)", "float64(float64, float64, float64, float64)"])
def _crps_tpnorm_ufunc(scale1: float, scale2: float, location: float, observation: float) -> float:
    crps1 = _crps_gtcnorm_ufunc(-float("inf"), 0, 0, scale2 / (scale1 + scale2), np.minimum(0, observation - location) / scale1)
    crps2 = _crps_gtcnorm_ufunc(0, float("inf"), scale2 / (scale1 + scale2), 0, np.maximum(0, observation - location) / scale2)
    out: float = scale1 * crps1 + scale2 * crps2
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
    "_crps_exponential_ufunc",
    "_crps_gamma_ufunc",
    "_crps_laplace_ufunc",
    "_crps_logistic_ufunc",
    "_crps_loglogistic_ufunc",
    "_crps_lognormal_ufunc",
    "_crps_normal_ufunc",
    "_crps_poisson_ufunc",
    "_crps_uniform_ufunc",
    "_crps_t_ufunc",
    "_crps_tpexponential_ufunc",
    "_crps_tpnorm_ufunc",
]
