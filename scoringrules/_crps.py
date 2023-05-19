from typing import Any, TypeVar

import numpy as np

from scoringrules.backend import backends as srb

Array = TypeVar("Array", np.ndarray, Any)
ArrayLike = TypeVar("ArrayLike", np.ndarray, Any, float)


def crps_ensemble(
    forecasts: Array,
    observations: ArrayLike,
    axis: int = -1,
    sorted_ensemble: bool = False,
    estimator: str = "pwm",
    backend: str = "numba",
) -> ArrayLike:
    r"""Estimate the Continuous Ranked Probability Score (CRPS) for a finite ensemble.

    Parameters
    ----------
    forecasts: ArrayLike
        The predicted forecast ensemble, where the ensemble dimension is by default
        represented by the last axis.
    observations: ArrayLike
        The observed values.
    axis: int
        The axis corresponding to the ensemble. Default is the last axis.
    sorted_ensemble: bool
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
    return srb[backend].crps_ensemble(
        forecasts,
        observations,
        axis=axis,
        sorted_ensemble=sorted_ensemble,
        estimator=estimator,
    )


def crps_normal(
    mu: ArrayLike, sigma: ArrayLike, observation: ArrayLike, backend: str = "numpy"
) -> ArrayLike:
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
    return srb[backend].crps_normal(mu, sigma, observation)


def crps_lognormal(
    mulog: ArrayLike,
    sigmalog: ArrayLike,
    observation: ArrayLike,
    backend: str = "numpy",
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
    return srb[backend].crps_lognormal(mulog, sigmalog, observation)


__all__ = [
    "crps_ensemble",
    "crps_normal",
    "crps_lognormal",
]
