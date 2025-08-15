import numpy as np
from numba import guvectorize

from scoringrules.core.utils import lazy_gufunc_wrapper_uv

INV_SQRT_PI = 1 / np.sqrt(np.pi)
EPSILON = 1e-6


@guvectorize("(),(n),(n)->()")
def _crps_ensemble_int_w_gufunc(
    obs: np.ndarray, fct: np.ndarray, w: np.ndarray, out: np.ndarray
):
    """CRPS estimator based on the integral form."""
    obs = obs[0]

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

        forecast_cdf += w[n]
        prev_forecast = forecast

    if obs_cdf == 0:
        integral += obs - forecast

    out[0] = integral


@guvectorize("(),(n),(n)->()")
def _crps_ensemble_qd_w_gufunc(
    obs: np.ndarray, fct: np.ndarray, w: np.ndarray, out: np.ndarray
):
    """CRPS estimator based on the quantile decomposition form."""
    obs = obs[0]

    if np.isnan(obs):
        out[0] = np.nan
        return

    obs_cdf = 0.0
    integral = 0.0
    a = np.cumsum(w) - w / 2

    for i, forecast in enumerate(fct):
        if obs < forecast:
            obs_cdf = 1.0

        integral += w[i] * (obs_cdf - a[i]) * (forecast - obs)

    out[0] = 2 * integral


@guvectorize("(),(n),(n)->()")
def _crps_ensemble_nrg_w_gufunc(
    obs: np.ndarray, fct: np.ndarray, w: np.ndarray, out: np.ndarray
):
    """CRPS estimator based on the energy form."""
    obs = obs[0]
    M = fct.shape[-1]

    if np.isnan(obs):
        out[0] = np.nan
        return

    e_1 = 0
    e_2 = 0

    for i in range(M):
        e_1 += abs(fct[i] - obs) * w[i]
        for j in range(i + 1, M):
            e_2 += abs(fct[j] - fct[i]) * w[j] * w[i]

    out[0] = e_1 - e_2


@guvectorize("(),(n),(n)->()")
def _crps_ensemble_fair_w_gufunc(
    obs: np.ndarray, fct: np.ndarray, w: np.ndarray, out: np.ndarray
):
    """Fair version of the CRPS estimator based on the energy form."""
    obs = obs[0]
    M = fct.shape[-1]

    if np.isnan(obs):
        out[0] = np.nan
        return

    e_1 = 0
    e_2 = 0

    for i in range(M):
        e_1 += abs(fct[i] - obs) * w[i]
        for j in range(i + 1, M):
            e_2 += abs(fct[j] - fct[i]) * w[j] * w[i]

    fair_c = 1 - np.sum(w**2)

    out[0] = e_1 - e_2 / fair_c


@guvectorize("(),(n),(n)->()")
def _crps_ensemble_pwm_w_gufunc(
    obs: np.ndarray, fct: np.ndarray, w: np.ndarray, out: np.ndarray
):
    """CRPS estimator based on the probability weighted moment (PWM) form."""
    obs = obs[0]

    if np.isnan(obs):
        out[0] = np.nan
        return

    expected_diff = 0.0
    β_0 = 0.0
    β_1 = 0.0

    w_sum = np.cumsum(w)

    for i, forecast in enumerate(fct):
        expected_diff += np.abs(forecast - obs) * w[i]
        β_0 += forecast * w[i] * (1.0 - w[i])
        β_1 += forecast * w[i] * (w_sum[i] - w[i])

    out[0] = expected_diff + β_0 - 2 * β_1


@guvectorize("(),(n),(n)->()")
def _crps_ensemble_akr_w_gufunc(
    obs: np.ndarray, fct: np.ndarray, w: np.ndarray, out: np.ndarray
):
    """CRPS estimaton based on the approximate kernel representation."""
    M = fct.shape[-1]
    obs = obs[0]
    e_1 = 0
    e_2 = 0
    for i, forecast in enumerate(fct):
        if i == 0:
            i = M - 1
        e_1 += abs(forecast - obs) * w[i]
        e_2 += abs(forecast - fct[i - 1]) * w[i]
    out[0] = e_1 - 0.5 * e_2


@guvectorize("(),(n),(n)->()")
def _crps_ensemble_akr_circperm_w_gufunc(
    obs: np.ndarray, fct: np.ndarray, w: np.ndarray, out: np.ndarray
):
    """CRPS estimaton based on the AKR with cyclic permutation."""
    M = fct.shape[-1]
    obs = obs[0]
    e_1 = 0.0
    e_2 = 0.0
    for i, forecast in enumerate(fct):
        sigma_i = int((i + 1 + ((M - 1) / 2)) % M)
        e_1 += abs(forecast - obs) * w[i]
        e_2 += abs(forecast - fct[sigma_i]) * w[i]
    out[0] = e_1 - 0.5 * e_2


@guvectorize("(),(n),(),(n),(n)->()")
def _owcrps_ensemble_nrg_w_gufunc(
    obs: np.ndarray,
    fct: np.ndarray,
    ow: np.ndarray,
    fw: np.ndarray,
    ens_w: np.ndarray,
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

    for i in range(M):
        e_1 += ens_w[i] * abs(fct[i] - obs) * fw[i] * ow
        for j in range(i + 1, M):
            e_2 += 2 * ens_w[i] * ens_w[j] * abs(fct[i] - fct[j]) * fw[i] * fw[j] * ow

    wbar = np.sum(ens_w * fw)

    out[0] = e_1 / wbar - 0.5 * e_2 / (wbar**2)


@guvectorize("(),(n),(),(n),(n)->()")
def _vrcrps_ensemble_nrg_w_gufunc(
    obs: np.ndarray,
    fct: np.ndarray,
    ow: np.ndarray,
    fw: np.ndarray,
    ens_w: np.ndarray,
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

    for i in range(M):
        e_1 += ens_w[i] * abs(fct[i] - obs) * fw[i] * ow
        for j in range(i + 1, M):
            e_2 += 2 * ens_w[i] * ens_w[j] * abs(fct[i] - fct[j]) * fw[i] * fw[j]

    wbar = np.sum(ens_w * fw)
    wabs_x = np.sum(ens_w * np.abs(fct) * fw)
    wabs_y = abs(obs) * ow

    out[0] = e_1 - 0.5 * e_2 + (wabs_x - wabs_y) * (wbar - ow)


estimator_gufuncs_w = {
    "akr_circperm": lazy_gufunc_wrapper_uv(_crps_ensemble_akr_circperm_w_gufunc),
    "akr": lazy_gufunc_wrapper_uv(_crps_ensemble_akr_w_gufunc),
    "fair": lazy_gufunc_wrapper_uv(_crps_ensemble_fair_w_gufunc),
    "int": lazy_gufunc_wrapper_uv(_crps_ensemble_int_w_gufunc),
    "nrg": lazy_gufunc_wrapper_uv(_crps_ensemble_nrg_w_gufunc),
    "pwm": lazy_gufunc_wrapper_uv(_crps_ensemble_pwm_w_gufunc),
    "qd": lazy_gufunc_wrapper_uv(_crps_ensemble_qd_w_gufunc),
    "ownrg": lazy_gufunc_wrapper_uv(_owcrps_ensemble_nrg_w_gufunc),
    "vrnrg": lazy_gufunc_wrapper_uv(_vrcrps_ensemble_nrg_w_gufunc),
}

__all__ = [
    "_crps_ensemble_akr_circperm_w_gufunc",
    "_crps_ensemble_akr_w_gufunc",
    "_crps_ensemble_fair_w_gufunc",
    "_crps_ensemble_int_w_gufunc",
    "_crps_ensemble_nrg_w_gufunc",
    "_crps_ensemble_pwm_w_gufunc",
    "_crps_ensemble_qd_w_gufunc",
]
