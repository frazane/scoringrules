import numpy as np
from scipy.stats import norm


def ensemble(pred, obs):
    """Energy form of the Continuous Ranked Probability Score (CRPS).

    $$a=1$$

    Parameters
    ----------
    pred: array-like
        The predicted forecast ensemble, where the ensemble dimension is represented by the first axis.
    obs: array_like
        The observed values.

    Returns
    -------
        crps: array_like
            The CRPS between the forecast ensemble and obs.


    Examples
    --------
    >>> from scoringrules import crps
    >>> crps.ensemble(pred, obs)
    """

    M = pred.shape[0]
    E_1 = np.sum(pred - obs[None, ...], axis=0) / M
    E_2 = np.sum(pred[None, ...] - pred[:, None, ...], axis=0) / (M * (M - 1))
    return E_1 - E_2 * 0.5


def normal(mu, sigma, obs):
    """Closed form of the Continuous Ranked Probability Score (CRPS) for the normal distribution.

    Parameters
    ----------
    mu: array_like
        Mean of the normal distribution.
    sigma: array_like
        Standard deviation of the normal distribution.

    Returns
    -------
    crps: array_like
        The CRPS between Normal(mu, sigma) and obs.
    """
    sx = (obs - mu) / sigma
    return sigma * (sx * (2 * norm.cdf(sx) - 1) + 2 * norm.pdf(sx) - 1 / np.sqrt(np.pi))


def lognormal(mu, sigma, obs):
    """Closed form of the Continuous Ranked Probability Score (CRPS) for the lognormal distribution.

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
    return obs * (2 * norm.cdf(sx) - 1) - ex * (
        norm.cdf(sx - sigma) + norm.cdf(sigma / np.sqrt(2)) - 1
    )
