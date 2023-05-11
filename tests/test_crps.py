import numpy as np
from scoringrules import crps

ENSEMBLE_SIZE = 51
N = 100


def test_ensemble_int():
    observation = np.random.randn(N)
    mu = observation + np.random.randn(N) * 0.1
    sigma = abs(np.random.randn(N)) * 0.3
    forecasts = np.random.randn(N, ENSEMBLE_SIZE) * sigma[..., None] + mu[..., None]

    # non-negative values
    res = crps.ensemble(forecasts, observation)
    assert not np.any(res < 0.0)

    # approx zero when perfect forecast
    perfect_forecasts = (
        observation[..., None] + np.random.randn(N, ENSEMBLE_SIZE) * 0.00001
    )
    res = crps.ensemble(perfect_forecasts, observation)
    assert not np.any(res - 0.0 > 0.0001)


def test_ensemble_gqf():
    observation = np.random.randn(N)
    mu = observation + np.random.randn(N) * 0.1
    sigma = abs(np.random.randn(N)) * 0.3
    forecasts = np.random.randn(N, ENSEMBLE_SIZE) * sigma[..., None] + mu[..., None]

    # non-negative values
    res = crps.ensemble(forecasts, observation, estimator="gqf")
    assert not np.any(res < 0.0)

    # approx zero when perfect forecast
    perfect_forecasts = (
        observation[..., None] + np.random.randn(N, ENSEMBLE_SIZE) * 0.00001
    )
    res = crps.ensemble(perfect_forecasts, observation, estimator="gqf")
    assert not np.any(res - 0.0 > 0.0001)


def test_normal():
    obs = np.random.randn(N)
    mu = obs + np.random.randn(N) * 0.1
    sigma = abs(np.random.randn(N)) * 0.3

    # non-negative values
    res = crps.normal(mu, sigma, obs)
    assert not np.any(res < 0.0)

    # approx zero when perfect forecast
    res = crps.normal(obs, 0.00001, obs)
    assert not np.any(res - 0.0 > 0.0001)


def test_lognormal():
    obs = np.random.randn(N)
    mu = obs + np.random.randn(N) * 0.1
    sigma = abs(np.random.randn(N)) * 0.3

    # non-negative values
    res = crps.normal(mu, sigma, obs)
    assert not np.any(res < 0.0)

    # approx zero when perfect forecast
    res = crps.normal(obs, 0.00001, obs)
    assert not np.any(res - 0.0 > 0.0001)
