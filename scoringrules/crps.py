import math

import numpy as np
from numba import guvectorize, njit, vectorize
from numpy.typing import ArrayLike

INV_SQRT_PI = 1 / np.sqrt(np.pi)


def ensemble(
    forecasts: ArrayLike,
    observations: ArrayLike,
    axis: int = -1,
    sorted_ensemble: bool = False,
    estimator: str = "int",
) -> ArrayLike:
    r"""Estimate the Continuous Ranked Probability Score (CRPS) for a finite ensemble.

    Parameters
    ----------
    forecasts: ArrayLike
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the last axis.
    observations: ArrayLike
        The observed values.
    axis: int, optional
        The axis corresponding to the ensemble. Default is the last axis.
    sorted_ensemble: bool, optional
        Boolean indicating whether the ensemble members are already in ascending order.
        Default is False.
    estimator: str
        Indicates the CRPS estimator to be used.

    Returns
    -------
    crps: ArrayLike
        The CRPS between the forecast ensemble and obs.

    Examples
    --------
    >>> from scoringrules import crps
    >>> crps.ensemble(pred, obs)
    """
    if axis != -1:
        forecasts = np.moveaxis(forecasts, axis, -1)

    if not sorted_ensemble and estimator != "nrg":
        forecasts = np.sort(forecasts, axis=-1)

    if estimator == "int":
        out = _crps_ensemble_int_gufunc(forecasts, observations)
    elif estimator == "qd":
        out = _crps_ensemble_qd_gufunc(forecasts, observations)
    elif estimator == "nrg":
        out = _crps_ensemble_nrg_gufunc(forecasts, observations)
    elif estimator == "pwm":
        out = _crps_ensemble_pwm_gufunc(forecasts, observations)
    elif estimator == "fair":
        out = _crps_ensemble_fair_gufunc(forecasts, observations)
    else:
        raise ValueError(f"{estimator} is not a valid estimator")

    return out


def normal(mu: ArrayLike, sigma: ArrayLike, observation: ArrayLike) -> ArrayLike:
    r"""Compute the closed form of the CRPS for the normal distribution.

    It is based on the following formulation from
    [Geiting et al. (2005)](https://journals.ametsoc.org/view/journals/mwre/133/5/mwr2904.1.xml):

    $$ \mathrm{CRPS}(\mathcal{N}(\mu, \sigma), y) = \sigma \Bigl\{ \omega [\Phi(ω) - 1] + 2 \phi(\omega) - \frac{1}{\sqrt{\pi}} \Bigl\},$$

    where $\Phi(ω)$ and $\phi(ω)$ are respectively the CDF and PDF of the standard normal
    distribution at the normalized prediction error $\omega = \frac{y - \mu}{\sigma}$.

    Parameters
    ----------
    mu: ArrayLike
        Mean of the forecast normal distribution.
    sigma: ArrayLike
        Standard deviation of the forecast normal distribution.
    observation: ArrayLike
        The observed values.

    Returns
    -------
    crps: array_like
        The CRPS between Normal(mu, sigma) and obs.

    Examples
    --------
    >>> from scoringrules import crps
    >>> crps.normal(0.1, 0.4, 0.0)
    """
    return _crps_normal_ufunc(mu, sigma, observation)


def lognormal(
    mulog: ArrayLike, sigmalog: ArrayLike, observation: ArrayLike
) -> ArrayLike:
    r"""Compute the closed form of the CRPS for the lognormal distribution.

    It is based on the formulation introduced by
    [Baran and Lerch (2015)](https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/qj.2521)

    $$ \mathrm{CRPS}(\mathrm{log}\mathcal{N}(\mu, \sigma), y) =
    y [2 \Phi(y) - 1] - 2 \mathrm{exp}(\mu + \frac{\sigma^2}{2})
    \left[ \Phi(\omega - \sigma) + \Phi(\frac{\sigma}{\sqrt{2}}) \right]$$

    where $\Phi$ is the CDF of the standard normal distribution and
    $\omega = \frac{\mathrm{log}y - \mu}{\sigma}$.


    Parameters
    ----------
    mulog: ArrayLike
        Mean of the normal underlying distribution.
    sigmalog: ArrayLike
        Standard deviation of the underlying normal distribution.

    Returns
    -------
    crps: ArrayLike
        The CRPS between Lognormal(mu, sigma) and obs.

    Notes
    -----
    The mean and standard deviation are not the values for the distribution itself,
    but of the underlying normal distribution it is derived from.

    Examples
    --------
    >>> from scoringrules import crps
    >>> crps.lognormal(0.1, 0.4, 0.0)
    """
    return _crps_lognormal_ufunc(mulog, sigmalog, observation)


@njit
def _norm_cdf(x):
    """Cumulative distribution function for the standard normal distribution."""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


@njit
def _norm_pdf(x):
    """Probability density function for the standard normal distribution."""
    return (1 / math.sqrt(2 * math.pi)) * math.exp(-(x**2) / 2)


@vectorize(["float32(float32, float32, float32)", "float64(float64, float64, float64)"])
def _crps_normal_ufunc(mu, sigma, observation):
    ω = (observation - mu) / sigma
    return sigma * (ω * (2 * _norm_cdf(ω) - 1) + 2 * _norm_pdf(ω) - INV_SQRT_PI)


@vectorize(["float32(float32, float32, float32)", "float64(float64, float64, float64)"])
def _crps_lognormal_ufunc(mulog, sigmalog, observation):
    ω = (np.log(observation) - mulog) / sigmalog
    ex = 2 * np.exp(mulog + sigmalog**2 / 2)
    return observation * (2 * _norm_cdf(ω) - 1) - ex * (
        _norm_cdf(ω - sigmalog) + _norm_cdf(sigmalog / np.sqrt(2)) - 1
    )


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:])",
    ],
    "(n),()->()",
)
def _crps_ensemble_int_gufunc(
    forecasts: ArrayLike, observation: ArrayLike, out: ArrayLike
) -> ArrayLike:
    """CRPS estimator based on the integral form."""
    obs = observation[0]
    M = forecasts.shape[0]

    if np.isnan(obs):
        out[0] = np.nan
        return

    obs_cdf = 0
    forecast_cdf = 0
    prev_forecast = 0
    integral = 0

    for n, forecast in enumerate(forecasts):
        if np.isnan(forecast):
            if n == 0:
                integral = np.nan
            forecast = prev_forecast
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
    forecasts: ArrayLike, observation: ArrayLike, out: ArrayLike
) -> ArrayLike:
    """CRPS estimator based on the quantile decomposition form."""
    obs = observation[0]
    M = forecasts.shape[0]

    if np.isnan(obs):
        out[0] = np.nan
        return

    obs_cdf = 0
    integral = 0

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
    forecasts: ArrayLike, observation: ArrayLike, out: ArrayLike
) -> ArrayLike:
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
    forecasts: ArrayLike, observation: ArrayLike, out: ArrayLike
) -> ArrayLike:
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
    forecasts: ArrayLike, observation: ArrayLike, out: ArrayLike
) -> ArrayLike:
    """CRPS estimator based on the probability weighted moment (PWM) form."""
    obs = observation[0]
    M = forecasts.shape[0]

    if np.isnan(obs):
        out[0] = np.nan
        return

    forecast_cdf = 0
    integral = 0
    obs_cdf = 0

    for n, forecast in enumerate(forecasts):
        if np.isnan(forecast):
            if n == 0:
                integral = np.nan
            break

        if obs_cdf == 0 and forecast > obs:
            obs_cdf = 1

        integral += 1 / M * np.abs(forecast - obs)
        integral += 1 / M * forecast
        integral -= 2 * (1 / (M - 1)) * forecast * forecast_cdf

        forecast_cdf += 1 / M

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
def _crps_ensemble_akr(forecasts, observation, out):
    """CRPS estimaton based on the approximate kernel representation."""
    M = forecasts.shape[-1]
    obs = observation[0]
    e_1 = 0
    e_2 = 0
    for i, forecast in enumerate(forecasts):
        if i == 0:
            continue
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
def _crps_ensemble_akr_circperm(forecasts, observation, out):
    """CRPS estimaton based on the AKR with cyclic permutation."""
    M = forecasts.shape[-1]
    obs = observation[0]
    sigma_i = lambda i: int((i + ((M - 1) / 2)) % M)
    e_1 = 0
    e_2 = 0
    for i, forecast in enumerate(forecasts):
        e_1 += abs(forecast - obs)
        e_2 += abs(forecast - forecasts[sigma_i(i + 1)])
    out[0] = e_1 / M - 0.5 * 1 / M * e_2


__all__ = [
    "ensemble",
    "normal",
    "lognormal",
]
