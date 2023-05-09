import numpy as np

from scoringutils import crps


def test_ensemble():
    M = 21
    obs = np.random.randn(10)
    pred = obs[None, ...] + np.random.randn(M, 10)
    res = crps.ensemble(pred, obs)

    pred_perfect = obs[None, ...] + np.random.randn(M, 10) * 1e-5
    res_zero = crps.ensemble(pred_perfect, obs)
    assert res_zero - 0.0 < 0.0001


def test_normal():
    obs = np.random.randn(10)
    mu, sigma = np.random.randn(10), abs(np.random.randn(10))
    res = crps.normal(mu, sigma, obs)

    res_zero = crps.normal(1, 0.00001, 1)
    assert res_zero - 0.0 < 0.0001


def test_normal():
    obs = np.random.randn(10)
    mu, sigma = np.random.randn(10), abs(np.random.randn(10))
    res = crps.normal(mu, sigma, obs)

    res_zero = crps.normal(1, 0.00001, 1)
    assert res_zero - 0.0 < 0.0001
