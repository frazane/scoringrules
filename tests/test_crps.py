import numpy as np
import pytest
from scoringrules import _crps

from .conftest import BACKENDS

ENSEMBLE_SIZE = 51
N = 100

ESTIMATORS = ["nrg", "fair", "pwm", "int", "qd", "akr", "akr_circperm"]


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_ensemble(estimator, backend):
    obs = np.random.randn(N)
    mu = obs + np.random.randn(N) * 0.1
    sigma = abs(np.random.randn(N)) * 0.3
    fct = np.random.randn(N, ENSEMBLE_SIZE) * sigma[..., None] + mu[..., None]

    # test exceptions
    if backend in ["numpy", "jax", "torch", "tensorflow"]:
        if estimator not in ["nrg", "fair", "pwm"]:
            with pytest.raises(ValueError):
                _crps.crps_ensemble(obs, fct, estimator=estimator, backend=backend)
            return

    # non-negative values
    res = _crps.crps_ensemble(obs, fct, estimator=estimator, backend=backend)
    res = np.asarray(res)
    assert not np.any(res < 0.0)

    # approx zero when perfect forecast
    perfect_fct = obs[..., None] + np.random.randn(N, ENSEMBLE_SIZE) * 0.00001
    res = _crps.crps_ensemble(obs, perfect_fct, estimator=estimator, backend=backend)
    res = np.asarray(res)
    assert not np.any(res - 0.0 > 0.0001)


@pytest.mark.parametrize("backend", BACKENDS)
def test_normal(backend):
    obs = np.random.randn(N)
    mu = obs + np.random.randn(N) * 0.1
    sigma = abs(np.random.randn(N)) * 0.3

    # non-negative values
    res = _crps.crps_normal(obs, mu, sigma, backend=backend)
    res = np.asarray(res)
    assert not np.any(np.isnan(res))
    assert not np.any(res < 0.0)

    # approx zero when perfect forecast
    mu = obs + np.random.randn(N) * 1e-6
    sigma = abs(np.random.randn(N)) * 1e-6
    res = _crps.crps_normal(obs, mu, sigma, backend=backend)
    res = np.asarray(res)

    assert not np.any(np.isnan(res))
    assert not np.any(res - 0.0 > 0.0001)


@pytest.mark.parametrize("backend", BACKENDS)
def test_lognormal(backend):
    obs = np.exp(np.random.randn(N))
    mulog = np.log(obs) + np.random.randn(N) * 0.1
    sigmalog = abs(np.random.randn(N)) * 0.3

    # non-negative values
    res = _crps.crps_lognormal(obs, mulog, sigmalog, backend=backend)
    res = np.asarray(res)
    assert not np.any(np.isnan(res))
    assert not np.any(res < 0.0)

    # approx zero when perfect forecast
    mulog = np.log(obs) + np.random.randn(N) * 1e-6
    sigmalog = abs(np.random.randn(N)) * 1e-6
    res = _crps.crps_lognormal(obs, mulog, sigmalog, backend=backend)
    res = np.asarray(res)

    assert not np.any(np.isnan(res))
    assert not np.any(res - 0.0 > 0.0001)
