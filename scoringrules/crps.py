import numpy as np
from numba import guvectorize
from numba_stats import norm
from numpy.typing import NDArray

INV_SQRT_PI = 1 / np.sqrt(np.pi)


def ensemble(
    forecasts: NDArray,
    observation: NDArray,
    axis: int = -1,
    sorted_ensemble: bool = False,
    estimator: str = "int",
):
    r"""Compute the Continuous Ranked Probability Score (CRPS).

    $$ \text{CRPS}_{\text{INT}}(M, y) = \int_{\mathbb{R}} \left[ \frac{1}{M}
    \sum_{i=1}^M \mathbb{1}\{x_i \le x \} - \mathbb{1}(y \le x)  \right] ^2 dx $$


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
    estimator: str
        Indicates the CRPS estimator to be used.

    Returns
    -------
    crps: NDArray
        The CRPS between the forecast ensemble and obs.

    Examples
    --------
    >>> from scoringrules import crps
    >>> crps.ensemble(pred, obs)
    """
    if not sorted_ensemble and estimator != "nrg":
        forecasts = np.sort(forecasts, axis=axis)

    if estimator == "int":
        out = _crps_ensemble_int_gufunc(forecasts, observation)
    elif estimator == "gqf":
        out = _crps_ensemble_gqf_gufunc(forecasts, observation)
    elif estimator == "nrg":
        out = _crps_ensemble_nrg_gufunc(forecasts, observation)
    else:
        raise ValueError("{estimator} is not a valid estimator")

    return out


def normal(mu: NDArray, sigma: NDArray, observation: NDArray) -> NDArray:
    r"""Compute the closed form of the CRPS for the normal distribution.

    It is based on the following formulation from
    [Geiting et al. (2005)](https://journals.ametsoc.org/view/journals/mwre/133/5/mwr2904.1.xml):

    $$ \mathrm{CRPS}(\mathcal{N}(\mu, \sigma), y) = \sigma \Bigl\{ \omega [\Phi(ω) - 1] + 2 \phi(\omega) - \frac{1}{\sqrt{\pi}} \Bigl\},$$

    where $\Phi(ω)$ and $\phi(ω)$ are respectively the CDF and PDF of the standard normal
    distribution at the normalized prediction error $\omega = \frac{y - \mu}{\sigma}$.

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

    Examples
    --------
    >>> from scoringrules import crps
    >>> crps.normal(0.1, 0.4, 0.0)
    """
    ω = (observation - mu) / sigma
    return sigma * (
        ω * (2 * norm.cdf(ω, 0, 1) - 1) + 2 * norm.pdf(ω, 0, 1) - INV_SQRT_PI
    )


def lognormal(mulog: NDArray, sigmalog: NDArray, observation: NDArray) -> NDArray:
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
    mulog: NDArray
        Mean of the normal underlying distribution.
    sigmalog: NDArray
        Standard deviation of the underlying normal distribution.

    Returns
    -------
    crps: NDArray
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
    ω = (np.log(obs) - mu) / sigma
    ex = 2 * np.exp(mu + sigma**2 / 2)
    crps = obs * (2 * norm.cdf(ω, 0, 1) - 1) - ex * (
        norm.cdf(ω - sigma, 0, 1) + norm.cdf(sigma / np.sqrt(2), 0, 1) - 1
    )
    return crps


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:])",
    ],
    "(n),()->()",
)
def _crps_ensemble_int_gufunc(forecasts, observation, result):
    """CRPS estimator based on the integral form."""
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


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:])",
    ],
    "(n),()->()",
)
def _crps_ensemble_gqf_gufunc(forecasts, observation, result):
    """CRPS estimator based on the generalized quantile function."""
    obs = observation[0]
    M = forecasts.shape[0]

    if np.isnan(obs):
        result[0] = np.nan
        return

    obs_cdf = 0
    integral = 0

    for i, forecast in enumerate(forecasts):
        if obs < forecast:
            obs_cdf = 1.0

        integral += (forecast - obs) * (M * obs_cdf - i + 0.5)

    result[0] = (2 / M**2) * integral


@guvectorize(
    [
        "void(float32[:], float32[:], float32[:])",
        "void(float64[:], float64[:], float64[:])",
    ],
    "(n),()->()",
)
def _crps_ensemble_nrg_gufunc(forecasts, observation, result):
    """CRPS estimator based on the energy form."""
    e_1 = np.nanmean(np.abs(observation - forecasts))
    e_2 = np.nanmean(
        np.abs(np.expand_dims(forecasts, 0) - np.expand_dims(forecasts, 1))
    )
    result[0] = e_1 - 0.5 * e_2


__all__ = [
    "ensemble",
    "normal",
    "lognormal",
]
