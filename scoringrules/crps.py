import numpy as np
from numba import guvectorize
from numpy.typing import NDArray
from scipy.stats import norm


def ensemble_int(
    forecasts: NDArray,
    observation: NDArray,
    axis: int = -1,
    sorted_ensemble: bool = False,
):
    r"""Compute the Continuous Ranked Probability Score (CRPS).

    $$ \text{CRPS}_{\text{INT}}(M, y) = \int_{\mathbb{R}} \left[ \frac{1}{M}
    \sum_{i=1}^M \mathbb{1}\{x_i \le x \} - \mathbb{1}(y \le x)  \right] ^2 dx$$


    Parameters
    ----------
    forecasts: NDArray
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the last axis.
    observation: NDArray
        The observed values.
    axis: int, optional
        The axis corresponding to the example. Default is the last axis.
    sorted_ensemble: bool, optional
        Boolean indicating whether the ensemble members are already in ascending order.
        Default is False.

    Returns
    -------
    crps: NDArray
        The CRPS between the forecast ensemble and obs.

    Examples
    --------
    >>> from scoringrules import crps
    >>> crps.ensemble(pred, obs)
    """
    if not sorted_ensemble:
        forecasts = np.sort(forecasts, axis=axis)

    return _ensemble_int_gufunc_alt(forecasts, observation)


def normal(mu: NDArray, sigma: NDArray, observation: NDArray):
    """Compute the closed form of the CRPS for the normal distribution.

    Parameters
    ----------
    mu: NDArray
        Mean of the forecast normal distribution.
    sigma: NDArray
        Standard deviation of the forecast normal distribution.
    observation: NDArray
        The observed values.

    Returns
    -------
    crps: array_like
        The CRPS between Normal(mu, sigma) and obs.
    """
    sx = (observation - mu) / sigma
    crps = sigma * (sx * (2 * norm.cdf(sx) - 1) + 2 * norm.pdf(sx) - 1 / np.sqrt(np.pi))
    return crps


def lognormal(mu, sigma, obs):
    """Compute the closed form of the CRPS for the lognormal distribution.

    Parameters
    ----------
    mu: array_like
        Mean of the normal distribution.
    sigma: array_like
        Standard deviation of the normal distribution.

    Returns
    -------
    crps: array_like
        The CRPS between Lognormal(mu, sigma) and obs.
    """
    sx = (np.log(obs) - mu) / sigma
    ex = 2 * np.exp(mu + sigma**2 / 2)
    crps = obs * (2 * norm.cdf(sx) - 1) - ex * (
        norm.cdf(sx - sigma) + norm.cdf(sigma / np.sqrt(2)) - 1
    )
    return crps


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:])",
    ],
    "(n),()->()",
)
def _ensemble_int_gufunc_alt(forecasts, observation, result):
    obs = observation[0]
    M = forecasts.shape[0]

    if np.isnan(obs):
        result[0] = np.nan
        return

    obs_cdf = 0
    forecast_cdf = 0
    prev_forecast = 0
    integral = 0

    for n, forecast in enumerate(forecasts):
        if np.isnan(forecast):
            # NumPy sorts NaN to the end
            if n == 0:
                integral = np.nan
            # reset for the sake of the conditional below
            forecast = prev_forecast
            break

        # this is evaluated the first time obs < forecast
        if obs_cdf == 0 and obs < forecast:
            integral += (obs - prev_forecast) * forecast_cdf**2
            integral += (forecast - obs) * (forecast_cdf - 1) ** 2
            obs_cdf = 1
        # this is the main integral  ∫F(x)-1(y<x)dx
        else:
            integral += (forecast_cdf - obs_cdf) ** 2 * (forecast - prev_forecast)

        forecast_cdf += 1 / M
        prev_forecast = forecast

    if obs_cdf == 0:
        integral += obs - forecast

    result[0] = integral


__all__ = [
    "ensemble_int",
    "normal",
    "lognormal",
]
