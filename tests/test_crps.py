import numpy as np
import pytest
import scipy.stats as st
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
def test_quantile_pinball(backend):
    # Test quantile approximation close to analytical normal crps if forecast comes from the normal distribution
    for mu in np.random.sample(size=10):
        for A in [9, 99, 999]:
            a0 = 1 / (A + 1)
            a1 = 1 - a0
            fct = (
                st.norm(np.repeat(mu, N), np.ones(N))
                .ppf(np.linspace(np.repeat(a0, N), np.repeat(a1, N), A))
                .T
            )
            alpha = np.linspace(a0, a1, A)
            obs = np.repeat(mu, N)
            percentage_error_to_analytic = 1 - _crps.crps_quantile(
                obs, fct, alpha, backend=backend
            ) / _crps.crps_normal(obs, mu, 1, backend=backend)
            percentage_error_to_analytic = np.asarray(percentage_error_to_analytic)
            assert np.all(
                np.abs(percentage_error_to_analytic) < 1 / A
            ), "Quantile CRPS should be close to normal CRPS"

    # Test raise valueerror if array sizes don't match
    with pytest.raises(ValueError):
        _crps.crps_quantile(obs, fct, alpha[0:42], backend=backend)
        return


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


@pytest.mark.parametrize("backend", BACKENDS)
def test_beta(backend):
    if backend == "torch":
        pytest.skip("Not implemented in torch backend")

    res = _crps.crps_beta(
        np.random.uniform(0, 1, (3, 3)),
        np.random.uniform(0, 3, (3, 3)),
        1.1,
        backend=backend,
    )
    assert res.shape == (3, 3)
    assert not np.any(np.isnan(res))

    # test exceptions
    with pytest.raises(ValueError):
        _crps.crps_beta(0.3, 0.7, 1.1, lower=1.0, upper=0.0, backend=backend)
        return

    # correctness tests
    res = _crps.crps_beta(0.3, 0.7, 1.1, backend=backend)
    expected = 0.0850102437
    assert np.isclose(res, expected)

    res = _crps.crps_beta(-3.0, 0.7, 1.1, lower=-5.0, upper=4.0, backend=backend)
    expected = 0.883206751
    assert np.isclose(res, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_binomial(backend):
    if backend == "torch":
        pytest.skip("Not implemented in torch backend")

    # test correctness
    res = _crps.crps_binomial(8, 10, 0.9, backend=backend)
    expected = 0.6685115
    assert np.isclose(res, expected)

    res = _crps.crps_binomial(-8, 10, 0.9, backend=backend)
    expected = 16.49896
    assert np.isclose(res, expected)

    res = _crps.crps_binomial(18, 10, 0.9, backend=backend)
    expected = 8.498957
    assert np.isclose(res, expected)

    # test broadcasting
    ones = np.ones(2)
    k, n, p = 8, 10, 0.9
    s = _crps.crps_binomial(k * ones, n, p, backend=backend)
    assert np.isclose(s, np.array([0.6685115, 0.6685115])).all()
    s = _crps.crps_binomial(k * ones, n * ones, p, backend=backend)
    assert np.isclose(s, np.array([0.6685115, 0.6685115])).all()
    s = _crps.crps_binomial(k * ones, n * ones, p * ones, backend=backend)
    assert np.isclose(s, np.array([0.6685115, 0.6685115])).all()
    s = _crps.crps_binomial(k, n * ones, p * ones, backend=backend)
    assert np.isclose(s, np.array([0.6685115, 0.6685115])).all()
    s = _crps.crps_binomial(k * ones, n, p * ones, backend=backend)
    assert np.isclose(s, np.array([0.6685115, 0.6685115])).all()


@pytest.mark.parametrize("backend", BACKENDS)
def test_exponentialM(backend):
    obs, mass, location, scale = 0.3, 0.1, 0.0, 1.0
    res = _crps.crps_exponentialM(obs, mass, location, scale, backend=backend)
    expected = 0.2384728
    assert np.isclose(res, expected)

    obs, mass, location, scale = 0.3, 0.1, -2.0, 3.0
    res = _crps.crps_exponentialM(obs, mass, location, scale, backend=backend)
    expected = 0.6236187
    assert np.isclose(res, expected)

    obs, mass, location, scale = -1.2, 0.1, -2.0, 3.0
    res = _crps.crps_exponentialM(obs, mass, location, scale, backend=backend)
    expected = 0.751013
    assert np.isclose(res, expected)
