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
    observation = np.random.randn(N)
    mu = observation + np.random.randn(N) * 0.1
    sigma = abs(np.random.randn(N)) * 0.3

    # non-negative values
    res = crps.normal(mu, sigma, observation)
    assert not np.any(res < 0.0)

    # approx zero when perfect forecast
    res = crps.normal(observation, 0.00001, observation)
    assert not np.any(res - 0.0 > 0.0001)


def test_lognormal():
    observation = np.random.lognormal(size=N)
    mu = observation + np.random.randn(N) * 0.1
    sigma = abs(np.random.randn(N)) * 0.3

    # non-negative values
    res = crps.lognormal(mu, sigma, observation)
    assert not np.any(res < 0.0)

    # approx zero when perfect forecast
    res = crps.lognormal(np.log(observation), 0.00001, observation)
    assert not np.any(res - 0.0 > 0.0001)
