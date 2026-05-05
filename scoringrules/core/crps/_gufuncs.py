import numpy as np
from numba import guvectorize

from scoringrules.core.utils import lazy_gufunc_wrapper_uv

INV_SQRT_PI = 1 / np.sqrt(np.pi)
EPSILON = 1e-6


@lazy_gufunc_wrapper_uv
@guvectorize("(),(a),(a)->()")
def quantile_pinball_gufunc(
    obs: np.ndarray, fct: np.ndarray, alpha: np.ndarray, out: np.ndarray
):
    """Pinball loss based estimator for the CRPS from quantiles."""
    Q = fct.shape[0]
    pb = 0
    for fc, level in zip(fct, alpha):  # noqa: B905
        pb += (obs > fc) * level * (obs - fc) + (obs <= fc) * (1 - level) * (fc - obs)
    out[0] = 2 * (pb / Q)


@guvectorize("(),(n)->()")
def _crps_ensemble_int_gufunc(obs: np.ndarray, fct: np.ndarray, out: np.ndarray):
    """CRPS estimator based on the integral form."""
    M = fct.shape[0]

    if np.isnan(obs):
        out[0] = np.nan
        return

    obs_cdf = 0
    forecast_cdf = 0.0
    prev_forecast = 0.0
    integral = 0.0

    for forecast in fct:
        if np.isnan(forecast):
            out[0] = np.nan
            return

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
        if np.isnan(forecast):
            out[0] = np.nan
            return

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

    for i in range(M):
        e_1 += abs(fct[i] - obs)
        for j in range(i + 1, M):
            e_2 += 2 * abs(fct[j] - fct[i])

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

    for i in range(M):
        e_1 += abs(fct[i] - obs)
        for j in range(i + 1, M):
            e_2 += 2 * abs(fct[j] - fct[i])

    out[0] = e_1 / M - 0.5 * e_2 / (M * (M - 1))


@guvectorize("(),(n)->()")
def _crps_ensemble_pwm_gufunc(obs: np.ndarray, fct: np.ndarray, out: np.ndarray):
    """CRPS estimator based on the probability weighted moment (PWM) form."""
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


@guvectorize("(),(n)->()")
def _crps_ensemble_akr_gufunc(obs: np.ndarray, fct: np.ndarray, out: np.ndarray):
    """CRPS estimaton based on the approximate kernel representation."""
    M = fct.shape[-1]
    e_1 = 0.0
    e_2 = 0.0
    for i, forecast in enumerate(fct):
        if i == 0:
            i = M
        e_1 += abs(forecast - obs)
        e_2 += abs(forecast - fct[i - 1])
    out[0] = e_1 / M - 0.5 * 1 / M * e_2


@guvectorize("(),(n)->()")
def _crps_ensemble_akr_circperm_gufunc(
    obs: np.ndarray, fct: np.ndarray, out: np.ndarray
):
    """CRPS estimaton based on the AKR with cyclic permutation."""
    M = fct.shape[-1]
    e_1 = 0.0
    e_2 = 0.0
    for i, forecast in enumerate(fct):
        sigma_i = int((i + 1 + ((M - 1) / 2)) % M)
        e_1 += abs(forecast - obs)
        e_2 += abs(forecast - fct[sigma_i])
    out[0] = e_1 / M - 0.5 * 1 / M * e_2


@guvectorize("(),(n),(),(n)->()")
def _owcrps_ensemble_nrg_gufunc(
    obs: np.ndarray,
    fct: np.ndarray,
    ow: np.ndarray,
    fw: np.ndarray,
    out: np.ndarray,
):
    """Outcome-weighted CRPS estimator based on the energy form."""
    M = fct.shape[-1]

    if np.isnan(obs):
        out[0] = np.nan
        return

    e_1 = 0.0
    e_2 = 0.0

    for i in range(M):
        e_1 += abs(fct[i] - obs) * fw[i] * ow
        for j in range(i + 1, M):
            e_2 += 2 * abs(fct[i] - fct[j]) * fw[i] * fw[j] * ow

    wbar = np.mean(fw)

    out[0] = e_1 / (M * wbar) - 0.5 * e_2 / ((M * wbar) ** 2)


@guvectorize("(),(n),(),(n)->()")
def _vrcrps_ensemble_nrg_gufunc(
    obs: np.ndarray,
    fct: np.ndarray,
    ow: np.ndarray,
    fw: np.ndarray,
    out: np.ndarray,
):
    """Vertically re-scaled CRPS estimator based on the energy form."""
    M = fct.shape[-1]

    if np.isnan(obs):
        out[0] = np.nan
        return

    e_1 = 0.0
    e_2 = 0.0

    for i in range(M):
        e_1 += abs(fct[i] - obs) * fw[i] * ow
        for j in range(i + 1, M):
            e_2 += 2 * abs(fct[i] - fct[j]) * fw[i] * fw[j]

    wbar = np.mean(fw)
    wabs_x = np.mean(np.abs(fct) * fw)
    wabs_y = abs(obs) * ow

    out[0] = e_1 / M - 0.5 * e_2 / (M**2) + (wabs_x - wabs_y) * (wbar - ow)


_estimator_gufuncs = {
    "akr_circperm": lazy_gufunc_wrapper_uv(_crps_ensemble_akr_circperm_gufunc),
    "akr": lazy_gufunc_wrapper_uv(_crps_ensemble_akr_gufunc),
    "fair": _crps_ensemble_fair_gufunc,
    "int": lazy_gufunc_wrapper_uv(_crps_ensemble_int_gufunc),
    "nrg": _crps_ensemble_nrg_gufunc,
    "pwm": lazy_gufunc_wrapper_uv(_crps_ensemble_pwm_gufunc),
    "qd": _crps_ensemble_qd_gufunc,
    "ownrg": lazy_gufunc_wrapper_uv(_owcrps_ensemble_nrg_gufunc),
    "vrnrg": lazy_gufunc_wrapper_uv(_vrcrps_ensemble_nrg_gufunc),
}


def estimator_gufuncs(estimator):
    if estimator not in _estimator_gufuncs:
        raise ValueError(f"Unknown estimator: {estimator}")
    return _estimator_gufuncs[estimator]


# --- NaN-omit variants ---
# Each gufunc below checks np.isnan inline and skips invalid members.  NaN
# values are left in-place by the public API (the zeroing done by
# apply_nan_policy_ens_uv is bypassed for the numba path), so the gufuncs must
# detect them themselves.
#
# For the sorted-ensemble estimators (qd, pwm, int) the public API layer
# pre-sorts the ensemble, which places NaN members at the tail (IEEE 754
# behaviour).  Those gufuncs exploit this with an early-exit loop.
# All other estimators (nrg, fair, akr, akr_circperm, ownrg, vrnrg) make no
# assumption about member order and scan the full array.


@guvectorize("(),(n)->()")
def _crps_ensemble_akr_nanomit_gufunc(
    obs: np.ndarray, fct: np.ndarray, out: np.ndarray
):
    """NaN-omit CRPS estimator based on the approximate kernel representation."""
    M = fct.shape[-1]
    e_1 = 0.0
    e_2 = 0.0
    M_eff = 0
    first_valid_val = 0.0
    prev_valid_val = 0.0

    for i in range(M):
        if np.isnan(fct[i]):
            continue
        M_eff += 1
        e_1 += abs(fct[i] - obs)
        if M_eff == 1:
            first_valid_val = fct[i]
        else:
            e_2 += abs(fct[i] - prev_valid_val)
        prev_valid_val = fct[i]

    # Circular wrap-around: pair last valid with first valid.
    if M_eff >= 2:
        e_2 += abs(first_valid_val - prev_valid_val)

    if M_eff == 0:
        out[0] = np.nan
    else:
        out[0] = e_1 / M_eff - 0.5 / M_eff * e_2


@guvectorize("(),(n)->()")
def _crps_ensemble_akr_circperm_nanomit_gufunc(
    obs: np.ndarray, fct: np.ndarray, out: np.ndarray
):
    """NaN-omit CRPS estimator based on the AKR with cyclic permutation."""
    M = fct.shape[-1]
    e_1 = 0.0
    e_2 = 0.0
    M_eff = 0

    # First pass: count valid members.
    for i in range(M):
        if not np.isnan(fct[i]):
            M_eff += 1

    if M_eff == 0:
        out[0] = np.nan
        return

    # Second pass: compute e_1 and e_2 using rank within valid members.
    i_eff = 0
    for i in range(M):
        if np.isnan(fct[i]):
            continue
        e_1 += abs(fct[i] - obs)
        sigma_i_eff = int((i_eff + 1 + ((M_eff - 1) / 2)) % M_eff)
        # Find the sigma_i_eff-th valid member.
        count = 0
        for j in range(M):
            if np.isnan(fct[j]):
                continue
            if count == sigma_i_eff:
                e_2 += abs(fct[i] - fct[j])
                break
            count += 1
        i_eff += 1

    out[0] = e_1 / M_eff - 0.5 / M_eff * e_2


estimator_gufuncs_nanomit = {
    "akr_circperm": lazy_gufunc_wrapper_uv(_crps_ensemble_akr_circperm_nanomit_gufunc),
    "akr": lazy_gufunc_wrapper_uv(_crps_ensemble_akr_nanomit_gufunc),
}
