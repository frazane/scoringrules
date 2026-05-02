import numpy as np
import pytest

import scoringrules as sr


ENSEMBLE_SIZE = 51
N = 100


def test_ensemble(backend):
    obs = np.random.randn(N)
    mu = obs + np.random.randn(N) * 0.1
    sigma = abs(np.random.randn(N))
    fct = np.random.randn(N, ENSEMBLE_SIZE) * sigma[..., None] + mu[..., None]

    res = sr.logs_ensemble(obs, fct, m_axis=-1, backend=backend)
    assert res.shape == (N,)

    fct = fct.T
    res0 = sr.logs_ensemble(obs, fct, m_axis=0, backend=backend)
    assert np.allclose(res, res0)

    obs, fct = 6.2, [4.2, 5.1, 6.1, 7.6, 8.3, 9.5]
    res = sr.logs_ensemble(obs, fct, backend=backend)
    expected = 1.926475
    assert np.isclose(res, expected)

    fct = [-142.7, -160.3, -179.4, -184.5]
    res = sr.logs_ensemble(obs, fct, backend=backend)
    expected = 55.25547
    assert np.isclose(res, expected)


def test_clogs(backend):
    obs = np.random.randn(N)
    mu = obs + np.random.randn(N) * 0.1
    sigma = abs(np.random.randn(N))
    fct = np.random.randn(N, ENSEMBLE_SIZE) * sigma[..., None] + mu[..., None]

    res0 = sr.logs_ensemble(obs, fct, m_axis=-1, backend=backend)
    res = sr.clogs_ensemble(obs, fct, m_axis=-1, backend=backend)
    res_co = sr.clogs_ensemble(obs, fct, m_axis=-1, cens=False, backend=backend)
    assert res.shape == (N,)
    assert res_co.shape == (N,)
    assert np.allclose(res, res0, atol=1e-5)
    assert np.allclose(res_co, res0, atol=1e-5)

    fct = fct.T
    res0 = sr.clogs_ensemble(obs, fct, m_axis=0, backend=backend)
    assert np.allclose(res, res0, atol=1e-5)

    obs, fct = 6.2, [4.2, 5.1, 6.1, 7.6, 8.3, 9.5]
    a = 8.0
    res = sr.clogs_ensemble(obs, fct, a=a, backend=backend)
    expected = 0.3929448
    assert np.isclose(res, expected)

    a = 5.0
    res = sr.clogs_ensemble(obs, fct, a=a, cens=False, backend=backend)
    expected = 1.646852
    assert np.isclose(res, expected)

    fct = [-142.7, -160.3, -179.4, -184.5]
    b = -150.0
    res = sr.clogs_ensemble(obs, fct, b=b, backend=backend)
    expected = 1.420427
    assert np.isclose(res, expected)


def test_logs_beta(backend):
    ## test shape

    # one observation, multiple forecasts
    res = sr.logs_beta(
        0.1,
        np.random.uniform(0, 3, (4, 3)),
        np.random.uniform(0, 3, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.logs_beta(
        np.random.uniform(0, 1, (4, 3)),
        2.4,
        0.5,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.logs_beta(
        np.random.uniform(0, 1, (4, 3)),
        np.random.uniform(0, 3, (4, 3)),
        np.random.uniform(0, 3, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # lower bound smaller than upper bound
    with pytest.raises(ValueError):
        sr.logs_beta(
            0.3, 0.7, 1.1, lower=1.0, upper=0.0, backend=backend, check_pars=True
        )
        return

    # lower bound equal to upper bound
    with pytest.raises(ValueError):
        sr.logs_beta(
            0.3, 0.7, 1.1, lower=0.5, upper=0.5, backend=backend, check_pars=True
        )
        return

    # negative shape parameters
    with pytest.raises(ValueError):
        sr.logs_beta(0.3, -0.7, 1.1, backend=backend, check_pars=True)
        return

    with pytest.raises(ValueError):
        sr.logs_beta(0.3, 0.7, -1.1, backend=backend, check_pars=True)
        return

    ## correctness tests

    # single forecast
    res = sr.logs_beta(0.3, 0.7, 1.1, backend=backend)
    expected = -0.04344567
    assert np.isclose(res, expected)

    # custom bounds
    res = sr.logs_beta(-3.0, 0.7, 1.1, lower=-5.0, upper=4.0, backend=backend)
    expected = 2.053211
    assert np.isclose(res, expected)

    # observation outside bounds
    res = sr.logs_beta(-3.0, 0.7, 1.1, lower=-1.3, upper=4.0, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    # multiple forecasts
    res = sr.logs_beta(
        -3.0,
        np.array([0.7, 2.0]),
        np.array([1.1, 4.8]),
        lower=np.array([-5.0, -5.0]),
        upper=np.array([4.0, 4.0]),
    )
    expected = np.array([2.053211, 1.329823])
    assert np.allclose(res, expected)


def test_logs_binomial(backend):
    if backend == "torch":
        pytest.skip("Not implemented in torch backend")

    ## test shape

    #    res = sr.logs_binomial(
    #        np.random.randint(0, 10, size=(4, 3)),
    #        np.full((4, 3), 10),
    #        np.random.uniform(0, 1, (4, 3)),
    #        backend=backend,
    #    )
    #    assert res.shape == (4, 3)
    #

    ones = np.ones(2)
    k, n, p = 8, 10, 0.9
    s = sr.logs_binomial(k * ones, n, p, backend=backend)
    assert np.isclose(s, np.array([1.64139182, 1.64139182])).all()
    s = sr.logs_binomial(k * ones, n * ones, p, backend=backend)
    assert np.isclose(s, np.array([1.64139182, 1.64139182])).all()
    s = sr.logs_binomial(k * ones, n * ones, p * ones, backend=backend)
    assert np.isclose(s, np.array([1.64139182, 1.64139182])).all()
    s = sr.logs_binomial(k, n * ones, p * ones, backend=backend)
    assert np.isclose(s, np.array([1.64139182, 1.64139182])).all()
    s = sr.logs_binomial(k * ones, n, p * ones, backend=backend)
    assert np.isclose(s, np.array([1.64139182, 1.64139182])).all()

    ## test exceptions

    # negative size parameter
    with pytest.raises(ValueError):
        sr.logs_binomial(7, -1, 0.9, backend=backend, check_pars=True)
        return

    # zero size parameter
    with pytest.raises(ValueError):
        sr.logs_binomial(2.1, 0, 0.1, backend=backend, check_pars=True)
        return

    # negative prob parameter
    with pytest.raises(ValueError):
        sr.logs_binomial(7, 15, -0.1, backend=backend, check_pars=True)
        return

    # prob parameter greater than one
    with pytest.raises(ValueError):
        sr.logs_binomial(1, 10, 1.1, backend=backend, check_pars=True)
        return

    # non-integer size parameter
    with pytest.raises(ValueError):
        sr.logs_binomial(4, 8.99, 0.5, backend=backend, check_pars=True)
        return

    ## test correctness

    # single observation
    res = sr.logs_binomial(8, 10, 0.9, backend=backend)
    expected = 1.641392
    assert np.isclose(res, expected)

    # negative observation
    res = sr.logs_binomial(-8, 10, 0.9, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    # zero prob parameter
    res = sr.logs_binomial(71, 212, 0, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    # observation larger than size
    res = sr.logs_binomial(18, 10, 0.9, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    # non-integer observation
    res = sr.logs_binomial(5.6, 10, 0.9, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    # multiple observations
    res = sr.logs_binomial(np.array([5.0, 17.2]), 10, 0.9, backend=backend)
    expected = np.array([6.510299, float("inf")])
    assert np.allclose(res, expected)


def test_logs_exponential(backend):
    ## test shape

    # one observation, multiple forecasts
    res = sr.logs_exponential(
        7.2,
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.logs_exponential(
        np.random.uniform(0, 10, (4, 3)),
        7.2,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.logs_exponential(
        np.random.uniform(0, 10, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # negative rate
    with pytest.raises(ValueError):
        sr.logs_exponential(3.2, -1, backend=backend, check_pars=True)
        return

    # zero rate
    with pytest.raises(ValueError):
        sr.logs_exponential(3.2, 0, backend=backend, check_pars=True)
        return

    ## test correctness

    # single observation
    res = sr.logs_exponential(3, 0.7, backend=backend)
    expected = 2.456675
    assert np.isclose(res, expected)

    # negative observation
    res = sr.logs_exponential(-3, 0.7, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    # multiple observations
    res = sr.logs_exponential(np.array([5.6, 17.2]), 10.1, backend=backend)
    expected = np.array([54.24746, 171.40746])
    assert np.allclose(res, expected)


def test_logs_2pexponential(backend):
    ## test shape

    # one observation, multiple forecasts
    res = sr.logs_2pexponential(
        6.2,
        np.random.uniform(0, 5, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        np.random.uniform(-2, 2, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.logs_2pexponential(
        np.random.uniform(-5, 5, (4, 3)),
        2.4,
        10.1,
        np.random.uniform(-2, 2, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.logs_2pexponential(
        np.random.uniform(-5, 5, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        np.random.uniform(-2, 2, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # negative scale parameters
    with pytest.raises(ValueError):
        sr.logs_2pexponential(3.2, -1, 2, 2.1, backend=backend, check_pars=True)
        return

    with pytest.raises(ValueError):
        sr.logs_2pexponential(3.2, 1, -2, 2.1, backend=backend, check_pars=True)
        return

    # zero scale parameters
    with pytest.raises(ValueError):
        sr.logs_2pexponential(3.2, 0, 2, 2.1, backend=backend, check_pars=True)
        return

    with pytest.raises(ValueError):
        sr.logs_2pexponential(3.2, 1, 0, 2.1, backend=backend, check_pars=True)
        return

    ## test correctness

    # single observation
    res = sr.logs_2pexponential(0.3, 0.1, 4.3, 0.0, backend=backend)
    expected = 1.551372
    assert np.isclose(res, expected)

    # negative location
    res = sr.logs_2pexponential(-20.8, 7.1, 2.0, -25.4, backend=backend)
    expected = 4.508274
    assert np.isclose(res, expected)

    # multiple observations
    res = sr.logs_2pexponential(
        np.array([-1.2, 3.7]), 0.1, 4.3, np.array([-0.1, 0.1]), backend=backend
    )
    expected = np.array([12.481605, 2.318814])
    assert np.allclose(res, expected)

    # check equivalence with laplace distribution
    res = sr.logs_2pexponential(-20.8, 2.0, 2.0, -25.4, backend=backend)
    res0 = sr.logs_laplace(-20.8, -25.4, 2.0, backend=backend)
    assert np.isclose(res, res0)


def test_logs_gamma(backend):
    ## test shape

    # one observation, multiple forecasts
    res = sr.logs_gamma(
        6.2,
        np.random.uniform(0, 5, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.logs_gamma(
        np.random.uniform(0, 10, (4, 3)),
        4.0,
        2.1,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.logs_gamma(
        np.random.uniform(0, 10, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # rate and scale provided
    with pytest.raises(ValueError):
        sr.logs_gamma(3.2, -1, rate=2, scale=0.4, backend=backend, check_pars=True)
        return

    # neither rate nor scale provided
    with pytest.raises(ValueError):
        sr.logs_gamma(3.2, -1, backend=backend, check_pars=True)
        return

    # negative shape parameter
    with pytest.raises(ValueError):
        sr.logs_gamma(3.2, -1, 2, backend=backend, check_pars=True)
        return

    # negative rate/scale parameter
    with pytest.raises(ValueError):
        sr.logs_gamma(3.2, 1, -2, backend=backend, check_pars=True)
        return

    # zero shape parameter
    with pytest.raises(ValueError):
        sr.logs_gamma(3.2, 0.0, 2, backend=backend, check_pars=True)
        return

    ## test correctness

    # single observation
    res = sr.logs_gamma(0.2, 1.1, 0.7, backend=backend)
    expected = 0.6434138
    assert np.isclose(res, expected)

    # scale instead of rate
    res = sr.logs_gamma(0.2, 1.1, scale=1 / 0.7, backend=backend)
    assert np.isclose(res, expected)

    # negative observation
    res = sr.logs_gamma(-4.2, 1.1, rate=0.7, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    # multiple observations
    res = sr.logs_gamma(np.array([-1.2, 2.3]), 1.1, 0.7, backend=backend)
    expected = np.array([float("inf"), 1.869179])
    assert np.allclose(res, expected)

    # check equivalence with exponential distribution
    res0 = sr.logs_exponential(3.1, 2, backend=backend)
    res = sr.logs_gamma(3.1, 1, 2, backend=backend)
    assert np.isclose(res0, res)


def test_logs_gev(backend):
    ## test shape

    # one observation, multiple forecasts
    res = sr.logs_gev(
        6.2,
        np.random.uniform(-10, 1, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        0.5,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.logs_gev(
        np.random.uniform(-5, 5, (4, 3)),
        -4.9,
        2.3,
        0.7,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.logs_gev(
        np.random.uniform(-5, 5, (4, 3)),
        np.random.uniform(-10, 1, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        0.7,
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # negative scale parameter
    with pytest.raises(ValueError):
        sr.logs_gev(0.7, 0.5, 2.0, -0.5, backend=backend, check_pars=True)
        return

    # zero scale parameter
    with pytest.raises(ValueError):
        sr.logs_gev(0.7, 0.5, 2.0, 0.0, backend=backend, check_pars=True)
        return

    ## test correctness

    # test default location
    res0 = sr.logs_gev(0.3, 0.0, backend=backend)
    res = sr.logs_gev(0.3, 0.0, 0.0, backend=backend)
    assert np.isclose(res0, res)

    # test default scale
    res = sr.logs_gev(0.3, 0.0, 0.0, 1.0, backend=backend)
    assert np.isclose(res0, res)

    # test invariance to shift
    mu = 0.1
    res = sr.logs_gev(0.3 + mu, 0.0, mu, 1.0, backend=backend)
    assert np.isclose(res0, res)

    # positive shape parameter
    res = sr.logs_gev(0.3, 0.7, backend=backend)
    expected = 1.22455
    assert np.isclose(res, expected)

    # negative shape parameter
    res = sr.logs_gev(0.3, -0.7, backend=backend)
    expected = 0.8151139
    assert np.isclose(res, expected)

    # zero shape parameter
    res = sr.logs_gev(0.3, 0.0, backend=backend)
    expected = 1.040818
    assert np.isclose(res, expected)

    # shape parameter greater than one
    res = sr.logs_gev(0.7, 1.5, 0.0, 0.5, backend=backend)
    expected = 1.662878
    assert np.isclose(res, expected)

    # multiple observations
    res = sr.logs_gev(np.array([0.9, -2.1]), -0.7, 0, 1.4, backend=backend)
    expected = np.array([1.018374, 2.817275])
    assert np.allclose(res, expected)


def test_logs_gpd(backend):
    ## test shape

    # one observation, multiple forecasts
    res = sr.logs_gpd(
        6.2,
        np.random.uniform(-10, 1, (4, 3)),
        np.random.uniform(-5, 5, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.logs_gpd(
        np.random.uniform(-5, 5, (4, 3)),
        -4.9,
        2.3,
        0.7,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.logs_gpd(
        np.random.uniform(-5, 5, (4, 3)),
        np.random.uniform(-10, 1, (4, 3)),
        np.random.uniform(-5, 5, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # negative scale parameter
    with pytest.raises(ValueError):
        sr.logs_gpd(0.7, 0.5, 2.0, -0.5, backend=backend, check_pars=True)
        return

    # zero scale parameter
    with pytest.raises(ValueError):
        sr.logs_gpd(0.7, 0.5, 2.0, 0.0, backend=backend, check_pars=True)
        return

    ## test correctness

    # test default location
    res0 = sr.logs_gpd(0.3, 0.0, backend=backend)
    res = sr.logs_gpd(0.3, 0.0, 0.0, backend=backend)
    assert np.isclose(res0, res)

    # test default scale
    res = sr.logs_gpd(0.3, 0.0, 0.0, 1.0, backend=backend)
    assert np.isclose(res0, res)

    # test invariance to shift
    res = sr.logs_gpd(0.3 + 0.1, 0.0, 0.1, 1.0, backend=backend)
    assert np.isclose(res0, res)

    # single observation
    res = sr.logs_gpd(0.3, 0.9, backend=backend)
    expected = 0.5045912
    assert np.isclose(res, expected)

    # negative observation
    res = sr.logs_gpd(-0.3, 0.9, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    # negative shape parameter
    res = sr.logs_gpd(0.3, -0.9, backend=backend)
    expected = 0.03496786
    assert np.isclose(res, expected)

    # shape parameter greater than one
    res = sr.logs_gpd(0.7, 1.5, 0.0, 0.5, backend=backend)
    expected = 1.192523
    assert np.isclose(res, expected)

    # multiple observations
    res = sr.logs_gpd(np.array([0.9, -2.1]), -0.7, 0, 1.4, backend=backend)
    expected = np.array([0.5926881, float("inf")])
    assert np.allclose(res, expected)


def test_logs_tlogis(backend):
    ## test shape

    # one observation, multiple forecasts
    res = sr.logs_tlogistic(
        17,
        np.random.uniform(10, 20, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.logs_tlogistic(
        np.random.uniform(-10, 10, (4, 3)),
        3.2,
        4.1,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.logs_tlogistic(
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # negative scale parameter
    with pytest.raises(ValueError):
        sr.logs_tlogistic(0.7, 2.0, -0.5, backend=backend, check_pars=True)
        return

    # zero scale parameter
    with pytest.raises(ValueError):
        sr.logs_tlogistic(0.7, 2.0, 0.0, backend=backend, check_pars=True)
        return

    # lower bound larger than upper bound
    with pytest.raises(ValueError):
        sr.logs_tlogistic(
            0.7, 2.0, 0.4, lower=-1, upper=-4, backend=backend, check_pars=True
        )
        return

    ## test correctness

    # default values
    res0 = sr.logs_tlogistic(1.8, backend=backend)
    res = sr.logs_tlogistic(1.8, 0.0, 1.0, -np.inf, np.inf, backend=backend)
    assert np.isclose(res0, res)

    # single observation
    res = sr.logs_tlogistic(1.8, -3.0, 3.3, -5.0, 4.7, backend=backend)
    expected = 2.485943
    assert np.isclose(res, expected)

    # observation below lower bound
    res = sr.logs_tlogistic(-5.8, -3.0, 3.3, -5.0, 4.7, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    # observation above upper bound
    res = sr.logs_tlogistic(7.8, -3.0, 3.3, -5.0, 4.7, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    # aligns with logs_logistic
    res0 = sr.logs_logistic(1.8, -3.0, 3.3, backend=backend)
    res = sr.logs_tlogistic(1.8, -3.0, 3.3, backend=backend)
    assert np.isclose(res, res0)

    # multiple observations
    res = sr.logs_tlogistic(
        np.array([1.8, -2.3]), 9.1, 11.1, -4.0, np.inf, backend=backend
    )
    expected = np.array([3.631568, 3.778196])
    assert np.allclose(res, expected)


def test_logs_tnormal(backend):
    ## test shape

    # one observation, multiple forecasts
    res = sr.logs_tnormal(
        17,
        np.random.uniform(10, 20, (4, 3)),
        np.random.uniform(0, 10, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.logs_tnormal(
        np.random.uniform(-10, 10, (4, 3)),
        3.2,
        4.1,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.logs_tnormal(
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(0, 10, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # negative scale parameter
    with pytest.raises(ValueError):
        sr.logs_tnormal(0.7, 2.0, -0.5, backend=backend, check_pars=True)
        return

    # zero scale parameter
    with pytest.raises(ValueError):
        sr.logs_tnormal(0.7, 2.0, 0.0, backend=backend, check_pars=True)
        return

    # lower bound larger than upper bound
    with pytest.raises(ValueError):
        sr.logs_tnormal(
            0.7, 2.0, 0.4, lower=-1, upper=-4, backend=backend, check_pars=True
        )
        return

    ## test correctness

    # default values
    res0 = sr.logs_tnormal(1.8, backend=backend)
    res = sr.logs_tnormal(1.8, 0.0, 1.0, -np.inf, np.inf, backend=backend)
    assert np.isclose(res0, res)

    # single observation
    res = sr.logs_tnormal(1.8, -3.0, 3.3, -5.0, 4.7, backend=backend)
    expected = 2.839353
    assert np.isclose(res, expected)

    # observation below lower bound
    res = sr.logs_tnormal(-5.8, -3.0, 3.3, -5.0, 4.7, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    # observation above upper bound
    res = sr.logs_tnormal(7.8, -3.0, 3.3, -5.0, 4.7, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    # aligns with logs_normal
    res0 = sr.logs_normal(1.8, -3.0, 3.3, backend=backend)
    res = sr.logs_tnormal(1.8, -3.0, 3.3, backend=backend)
    assert np.isclose(res, res0)

    # multiple observations
    res = sr.logs_tnormal(
        np.array([1.8, -2.3]), 9.1, 11.1, -4.0, np.inf, backend=backend
    )
    expected = np.array([3.415483, 3.726619])
    assert np.allclose(res, expected)


def test_logs_tt(backend):
    if backend in ["jax", "torch", "tensorflow"]:
        pytest.skip("Not implemented in jax, torch or tensorflow backends")

    ## test shape

    # one observation, multiple forecasts
    res = sr.logs_tt(
        17,
        np.random.uniform(0, 20, (4, 3)),
        np.random.uniform(10, 20, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.logs_tt(
        np.random.uniform(-10, 10, (4, 3)),
        11.1,
        3.2,
        4.1,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.logs_tt(
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(0, 20, (4, 3)),
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # negative df parameter
    with pytest.raises(ValueError):
        sr.logs_tt(0.7, -1.7, 2.0, 0.5, backend=backend, check_pars=True)
        return

    # zero df parameter
    with pytest.raises(ValueError):
        sr.logs_tt(0.7, 0.0, 2.0, 0.5, backend=backend, check_pars=True)
        return

    # negative scale parameter
    with pytest.raises(ValueError):
        sr.logs_tt(0.7, 4.7, 2.0, -0.5, backend=backend, check_pars=True)
        return

    # zero scale parameter
    with pytest.raises(ValueError):
        sr.logs_tt(0.7, 4.7, 2.0, 0.0, backend=backend, check_pars=True)
        return

    # lower bound larger than upper bound
    with pytest.raises(ValueError):
        sr.logs_tt(
            0.7, 4.7, 2.0, 0.4, lower=-1, upper=-4, backend=backend, check_pars=True
        )
        return

    ## test correctness

    # default values
    res0 = sr.logs_tt(1.8, 6.9, backend=backend)
    res = sr.logs_tt(1.8, 6.9, 0.0, 1.0, -np.inf, np.inf, backend=backend)
    assert np.isclose(res0, res)

    # single observation
    res = sr.logs_tt(1.8, 6.9, -3.0, 3.3, -5.0, 4.7, backend=backend)
    expected = 2.836673
    assert np.isclose(res, expected)

    # observation below lower bound
    res = sr.logs_tt(-5.8, 6.9, -3.0, 3.3, -5.0, 4.7, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    # observation above upper bound
    res = sr.logs_tt(7.8, 6.9, -3.0, 3.3, -5.0, 4.7, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    # aligns with logs_t
    res0 = sr.logs_t(1.8, 6.9, -3.0, 3.3, backend=backend)
    res = sr.logs_tt(1.8, 6.9, -3.0, 3.3, backend=backend)
    assert np.isclose(res, res0)

    # multiple observations
    res = sr.logs_tt(
        np.array([1.8, -2.3]), 6.9, 9.1, 11.1, -4.0, np.inf, backend=backend
    )
    expected = np.array([3.453054, 3.774802])
    assert np.allclose(res, expected)


def test_logs_hypergeometric(backend):
    if backend == "torch":
        pytest.skip("Currently not working in torch backend")

    ## test shape

    # one observation, multiple forecasts
    res = sr.logs_hypergeometric(
        17,
        np.random.randint(10, 20, size=(4, 3)),
        np.random.randint(10, 20, size=(4, 3)),
        np.random.randint(1, 10, size=(4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.logs_hypergeometric(
        np.random.randint(0, 20, size=(4, 3)),
        13,
        4,
        7,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.logs_hypergeometric(
        np.random.randint(0, 20, size=(4, 3)),
        np.random.randint(10, 20, size=(4, 3)),
        np.random.randint(10, 20, size=(4, 3)),
        np.random.randint(1, 10, size=(4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # negative m parameter
    with pytest.raises(ValueError):
        sr.logs_hypergeometric(7, m=-1, n=10, k=5, backend=backend, check_pars=True)
        return

    # negative n parameter
    with pytest.raises(ValueError):
        sr.logs_hypergeometric(7, m=11, n=-2, k=5, backend=backend, check_pars=True)
        return

    # negative k parameter
    with pytest.raises(ValueError):
        sr.logs_hypergeometric(7, m=11, n=10, k=-5, backend=backend, check_pars=True)
        return

    # k > m + n
    with pytest.raises(ValueError):
        sr.logs_hypergeometric(7, m=11, n=10, k=25, backend=backend, check_pars=True)
        return

    # non-integer m parameter
    with pytest.raises(ValueError):
        sr.logs_hypergeometric(7, m=11.1, n=10, k=5, backend=backend, check_pars=True)
        return

    # non-integer n parameter
    with pytest.raises(ValueError):
        sr.logs_hypergeometric(7, m=11, n=10.8, k=5, backend=backend, check_pars=True)
        return

    # non-integer k parameter
    with pytest.raises(ValueError):
        sr.logs_hypergeometric(7, m=11, n=10, k=4.3, backend=backend, check_pars=True)
        return

    ## test correctness

    # single observation
    res = sr.logs_hypergeometric(5, 7, 13, 12, backend=backend)
    expected = 1.251525
    assert np.isclose(res, expected)

    # negative observation
    res = sr.logs_hypergeometric(-5, 7, 13, 12, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    # non-integer observation
    res = sr.logs_hypergeometric(5.6, 7, 13, 12, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    # multiple observations
    res = sr.logs_hypergeometric(np.array([2, 7.6, -2.1]), 7, 13, 3, backend=backend)
    expected = np.array([1.429312, float("inf"), float("inf")])
    assert np.allclose(res, expected)


def test_logs_laplace(backend):
    ## test shape

    # one observation, multiple forecasts
    res = sr.logs_laplace(
        17,
        np.random.uniform(10, 20, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.logs_laplace(
        np.random.uniform(-10, 10, (4, 3)),
        3.2,
        4.1,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.logs_laplace(
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # negative scale parameter
    with pytest.raises(ValueError):
        sr.logs_laplace(7, 4, -0.1, backend=backend, check_pars=True)
        return

    # zero scale parameter
    with pytest.raises(ValueError):
        sr.logs_laplace(7, 4, 0.0, backend=backend, check_pars=True)
        return

    ## test correctness

    # default location
    res0 = sr.logs_laplace(-3, backend=backend)
    res = sr.logs_laplace(-3, location=0, backend=backend)
    assert np.isclose(res0, res)

    # default scale
    res = sr.logs_laplace(-3, location=0, scale=1.0, backend=backend)
    assert np.isclose(res0, res)

    # invariance to location shifts
    res = sr.logs_laplace(-3 + 0.1, location=0.1, scale=1.0, backend=backend)
    assert np.isclose(res0, res)

    # single observation
    res = sr.logs_laplace(-2.9, 0.1, backend=backend)
    expected = 3.693147
    assert np.isclose(res, expected)

    # multiple observations
    res = sr.logs_laplace(
        np.array([-2.9, 4.1]),
        np.array([1.1, 1.8]),
        np.array([10.1, 1.2]),
        backend=backend,
    )
    expected = np.array([3.401722, 2.792135])
    assert np.allclose(res, expected)


def test_logs_logis(backend):
    ## test shape

    # one observation, multiple forecasts
    res = sr.logs_logistic(
        17,
        np.random.uniform(10, 20, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.logs_logistic(
        np.random.uniform(-10, 10, (4, 3)),
        3.2,
        4.1,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.logs_logistic(
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # negative scale parameter
    with pytest.raises(ValueError):
        sr.logs_logistic(7, 4, -0.1, backend=backend, check_pars=True)
        return

    # zero scale parameter
    with pytest.raises(ValueError):
        sr.logs_logistic(7, 4, 0.0, backend=backend, check_pars=True)
        return

    ## test correctness

    # default location
    res0 = sr.logs_logistic(-3, backend=backend)
    res = sr.logs_logistic(-3, location=0, backend=backend)
    assert np.isclose(res0, res)

    # default scale
    res = sr.logs_logistic(-3, location=0, scale=1.0, backend=backend)
    assert np.isclose(res0, res)

    # invariance to location shifts
    res = sr.logs_logistic(-3 + 0.1, location=0.1, scale=1.0, backend=backend)
    assert np.isclose(res0, res)

    # single observation
    res = sr.logs_logistic(17.1, 13.8, 3.3, backend=backend)
    expected = 2.820446
    assert np.isclose(res, expected)

    res = sr.logs_logistic(3.1, 4.0, 0.5, backend=backend)
    expected = 1.412808
    assert np.isclose(res, expected)

    # multiple observations
    res = sr.logs_logistic(
        np.array([-2.9, 4.1]),
        np.array([1.1, 1.8]),
        np.array([10.1, 1.2]),
        backend=backend,
    )
    expected = np.array([3.737788, 2.373456])
    assert np.allclose(res, expected)


def test_logs_loglaplace(backend):
    ## test shape

    # one observation, multiple forecasts
    res = sr.logs_loglaplace(
        17,
        np.random.uniform(10, 20, (4, 3)),
        np.random.uniform(0, 1, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.logs_loglaplace(
        np.random.uniform(-10, 10, (4, 3)),
        3.2,
        0.9,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.logs_loglaplace(
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(0, 1, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # negative scale parameter
    with pytest.raises(ValueError):
        sr.logs_loglaplace(7, 4, -0.1, backend=backend, check_pars=True)
        return

    # zero scale parameter
    with pytest.raises(ValueError):
        sr.logs_loglaplace(7, 4, 0.0, backend=backend, check_pars=True)
        return

    # scale parameter larger than one
    with pytest.raises(ValueError):
        sr.logs_loglaplace(7, 4, 1.1, backend=backend, check_pars=True)
        return

    ## test correctness

    # single observation
    res = sr.logs_loglaplace(7.1, 3.8, 0.3, backend=backend)
    expected = 7.582287
    assert np.isclose(res, expected)

    # negative observation
    res = sr.logs_loglaplace(-4.1, -3.8, 0.3, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    # multiple observations
    res = sr.logs_loglaplace(
        np.array([2.9, 4.4]),
        np.array([1.1, 1.8]),
        np.array([0.1, 0.9]),
        backend=backend,
    )
    expected = np.array([-0.1918345, 2.4231639])
    assert np.allclose(res, expected)


def test_logs_loglogistic(backend):
    ## test shape

    # one observation, multiple forecasts
    res = sr.logs_loglogistic(
        17,
        np.random.uniform(10, 20, (4, 3)),
        np.random.uniform(0, 1, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.logs_loglogistic(
        np.random.uniform(-10, 10, (4, 3)),
        3.2,
        0.9,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.logs_loglogistic(
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(0, 1, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # negative scale parameter
    with pytest.raises(ValueError):
        sr.logs_loglogistic(7, 4, -0.1, backend=backend, check_pars=True)
        return

    # zero scale parameter
    with pytest.raises(ValueError):
        sr.logs_loglogistic(7, 4, 0.0, backend=backend, check_pars=True)
        return

    # scale parameter larger than one
    with pytest.raises(ValueError):
        sr.logs_loglogistic(7, 4, 1.1, backend=backend, check_pars=True)
        return

    ## test correctness

    # single observation
    res = sr.logs_loglogistic(7.1, 3.8, 0.3, backend=backend)
    expected = 6.893475
    assert np.isclose(res, expected)

    # negative observation
    res = sr.logs_loglogistic(-4.1, -3.8, 0.3, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    # multiple observations
    res = sr.logs_loglogistic(
        np.array([2.9, 4.4]),
        np.array([1.1, 1.8]),
        np.array([0.1, 0.9]),
        backend=backend,
    )
    expected = np.array([0.1793931, 2.7936654])
    assert np.allclose(res, expected)


def test_logs_lognormal(backend):
    ## test shape

    # one observation, multiple forecasts
    res = sr.logs_lognormal(
        17,
        np.random.uniform(10, 20, (4, 3)),
        np.random.uniform(0, 10, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.logs_lognormal(
        np.random.uniform(0, 10, (4, 3)),
        3.2,
        2.9,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.logs_lognormal(
        np.random.uniform(0, 10, (4, 3)),
        np.random.uniform(0, 10, (4, 3)),
        np.random.uniform(0, 1, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # negative scale parameter
    with pytest.raises(ValueError):
        sr.logs_lognormal(7, 4, -0.1, backend=backend, check_pars=True)
        return

    # zero scale parameter
    with pytest.raises(ValueError):
        sr.logs_lognormal(7, 4, 0.0, backend=backend, check_pars=True)
        return

    ## test correctness

    # single observation
    res = sr.logs_lognormal(7.1, 3.8, 0.3, backend=backend)
    expected = 20.48201
    assert np.isclose(res, expected)

    # negative observation
    res = sr.logs_lognormal(-4.1, -3.8, 0.3, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    # multiple observations
    res = sr.logs_lognormal(
        np.array([2.9, 4.4]),
        np.array([1.1, 1.8]),
        np.array([0.1, 0.9]),
        backend=backend,
    )
    expected = np.array([-0.2566692, 2.3577601])
    assert np.allclose(res, expected)


def test_logs_mixnorm(backend):
    ## test shape

    # one observation, multiple forecasts
    res = sr.logs_mixnorm(
        17,
        np.random.uniform(-5, 5, (4, 3, 5)),
        np.random.uniform(0, 5, (4, 3, 5)),
        np.random.uniform(0, 1, (4, 3, 5)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.logs_mixnorm(
        np.random.uniform(-5, 5, (4, 3)),
        np.array([0.0, -2.9, 0.9, 3.2, 7.0]),
        np.array([0.6, 2.9, 0.9, 3.2, 7.0]),
        np.array([0.6, 2.9, 0.9, 0.2, 0.1]),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.logs_mixnorm(
        np.random.uniform(-5, 5, (4, 3)),
        np.random.uniform(-5, 5, (4, 3, 5)),
        np.random.uniform(0, 5, (4, 3, 5)),
        np.random.uniform(0, 1, (4, 3, 5)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # negative scale parameter
    with pytest.raises(ValueError):
        sr.logs_mixnorm(
            7, np.array([4, 2]), np.array([-1, 2]), backend=backend, check_pars=True
        )
        return

    # negative weight parameter
    with pytest.raises(ValueError):
        sr.logs_mixnorm(
            7,
            np.array([4, 2]),
            np.array([1, 2]),
            np.array([-1, 2]),
            backend=backend,
            check_pars=True,
        )
        return

    ## test correctness

    # single observation
    obs, m, s, w = 0.3, [0.0, -2.9, 0.9], [0.5, 1.4, 0.7], [1 / 3, 1 / 3, 1 / 3]
    res = sr.logs_mixnorm(obs, m, s, w, backend=backend)
    expected = 1.019742
    assert np.isclose(res, expected)

    # default weights
    res0 = sr.logs_mixnorm(obs, m, s, backend=backend)
    assert np.isclose(res, res0)

    # non-constant weights
    w = [0.3, 0.1, 0.6]
    res = sr.logs_mixnorm(obs, m, s, w, backend=backend)
    expected = 0.8235977
    assert np.isclose(res, expected)

    # weights that do not sum to one
    w = [8.3, 0.1, 4.6]
    res = sr.logs_mixnorm(obs, m, s, w, backend=backend)
    expected = 0.5703479
    assert np.isclose(res, expected)

    # m-axis argument
    obs = [-1.6, 0.3]
    m = [[0.0, -2.9], [0.6, 0.0], [-1.1, -2.3]]
    s = [[0.5, 1.7], [1.1, 0.7], [1.4, 1.5]]
    res1 = sr.logs_mixnorm(obs, m, s, m_axis=0, backend=backend)

    m = [[0.0, 0.6, -1.1], [-2.9, 0.0, -2.3]]
    s = [[0.5, 1.1, 1.4], [1.7, 0.7, 1.5]]
    res2 = sr.logs_mixnorm(obs, m, s, backend=backend)
    assert np.allclose(res1, res2)

    # check equality with normal distribution
    obs = 1.6
    m, s, w = [1.8, 1.8], [2.3, 2.3], [0.6, 0.8]
    res0 = sr.logs_normal(obs, m[0], s[0], backend=backend)
    res = sr.logs_mixnorm(obs, m, s, w, backend=backend)
    assert np.isclose(res0, res)


def test_logs_negbinom(backend):
    if backend in ["jax", "torch", "tensorflow"]:
        pytest.skip("Not implemented in jax, torch or tensorflow backends")

    ## test shape

    # one observation, multiple forecasts
    res = sr.logs_negbinom(
        17,
        np.random.uniform(10, 20, (4, 3)),
        np.random.uniform(0, 1, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.logs_negbinom(
        np.random.uniform(0, 20, (4, 3)),
        12,
        0.2,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.logs_negbinom(
        np.random.uniform(0, 20, (4, 3)),
        np.random.uniform(10, 20, (4, 3)),
        np.random.uniform(0, 1, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # prob and mu are both provided
    with pytest.raises(ValueError):
        sr.logs_negbinom(0.3, 7.0, 0.8, mu=7.3, backend=backend)

    # neither prob nor mu are provided
    with pytest.raises(ValueError):
        sr.logs_negbinom(0.3, 8, backend=backend)

    # negative size parameter
    with pytest.raises(ValueError):
        sr.logs_negbinom(4.3, -8, 0.8, backend=backend, check_pars=True)

    # negative mu parameter
    with pytest.raises(ValueError):
        sr.logs_negbinom(4.3, 8, mu=-0.8, backend=backend, check_pars=True)

    # negative prob parameter
    with pytest.raises(ValueError):
        sr.logs_negbinom(4.3, 8, -0.8, backend=backend, check_pars=True)

    # prob parameter larger than one
    with pytest.raises(ValueError):
        sr.logs_negbinom(4.3, 8, 1.8, backend=backend, check_pars=True)

    ## test correctness

    # single observation
    res = sr.logs_negbinom(2.0, 7.0, 0.8, backend=backend)
    expected = 1.448676
    assert np.isclose(res, expected)

    # non-integer observation
    res = sr.logs_negbinom(1.5, 2.4, 0.5, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    # negative observation
    res = sr.logs_negbinom(-1.0, 17.0, 0.1, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    # mu given instead of prob
    res = sr.logs_negbinom(2.0, 11.0, mu=7.3, backend=backend)
    expected = 3.247462
    assert np.isclose(res, expected)

    # multiple observations
    res = sr.logs_negbinom(np.array([2.0, 7.0]), 7.0, 0.8, backend=backend)
    expected = np.array([1.448676, 5.380319])
    assert np.allclose(res, expected)


def test_logs_normal(backend):
    ## test shape

    # one observation, multiple forecasts
    res = sr.logs_normal(
        17,
        np.random.uniform(10, 20, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.logs_normal(
        np.random.uniform(-10, 10, (4, 3)),
        3.2,
        4.1,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.logs_normal(
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(0, 10, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # negative scale parameter
    with pytest.raises(ValueError):
        sr.logs_normal(7, 4, -0.1, backend=backend, check_pars=True)
        return

    # zero scale parameter
    with pytest.raises(ValueError):
        sr.logs_normal(7, 4, 0.0, backend=backend, check_pars=True)
        return

    ## test correctness

    # default mu
    res0 = sr.logs_normal(-3, backend=backend)
    res = sr.logs_normal(-3, mu=0, backend=backend)
    assert np.isclose(res0, res)

    # default sigma
    res = sr.logs_normal(-3, mu=0, sigma=1.0, backend=backend)
    assert np.isclose(res0, res)

    # invariance to location shifts
    res = sr.logs_normal(-3 + 0.1, mu=0.1, sigma=1.0, backend=backend)
    assert np.isclose(res0, res)

    # single observation
    res = sr.logs_normal(7.1, 3.8, 0.3, backend=backend)
    expected = 60.21497
    assert np.isclose(res, expected)

    res = sr.logs_normal(3.1, 4.0, 0.5, backend=backend)
    expected = 1.845791
    assert np.isclose(res, expected)

    # multiple observations
    res = sr.logs_normal(
        np.array([-2.9, 4.4]),
        np.array([1.1, 1.8]),
        np.array([10.1, 1.2]),
        backend=backend,
    )
    expected = np.array([3.309898, 3.448482])
    assert np.allclose(res, expected)


def test_logs_2pnormal(backend):
    ## test shape

    # one observation, multiple forecasts
    res = sr.logs_2pnormal(
        17,
        np.random.uniform(10, 20, (4, 3)),
        np.random.uniform(0, 10, (4, 3)),
        np.random.uniform(0, 10, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.logs_2pnormal(
        np.random.uniform(-5, 5, (4, 3)),
        3.2,
        4.1,
        2.8,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.logs_2pnormal(
        np.random.uniform(-5, 5, (4, 3)),
        np.random.uniform(-5, 5, (4, 3)),
        np.random.uniform(0, 10, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # negative scale1 parameter
    with pytest.raises(ValueError):
        sr.logs_2pnormal(7, -0.1, 0.1, backend=backend, check_pars=True)
        return

    # zero scale1 parameter
    with pytest.raises(ValueError):
        sr.logs_2pnormal(7, 0.0, backend=backend, check_pars=True)
        return

    # negative scale2 parameter
    with pytest.raises(ValueError):
        sr.logs_2pnormal(7, 0.1, -2.1, backend=backend, check_pars=True)
        return

    # zero scale2 parameter
    with pytest.raises(ValueError):
        sr.logs_2pnormal(7, 1.1, 0.0, backend=backend, check_pars=True)
        return

    ## test correctness

    # default scale1
    res0 = sr.logs_2pnormal(-3, backend=backend)
    res = sr.logs_2pnormal(-3, scale1=1.0, backend=backend)
    assert np.isclose(res0, res)

    # default scale2
    res = sr.logs_2pnormal(-3, scale1=1.0, scale2=1.0, backend=backend)
    assert np.isclose(res0, res)

    # default location
    res = sr.logs_2pnormal(-3, scale1=1.0, scale2=1.0, location=0.0, backend=backend)
    assert np.isclose(res0, res)

    # invariance to location shifts
    res = sr.logs_2pnormal(-3 + 0.1, 1.0, 1.0, location=0.1, backend=backend)
    assert np.isclose(res0, res)

    # single observation
    res = sr.logs_2pnormal(7.1, 3.8, 1.3, 2.0, backend=backend)
    expected = 9.550298
    assert np.isclose(res, expected)

    # check equivalence with standard normal distribution
    res0 = sr.logs_normal(3.1, 4.0, 0.5, backend=backend)
    res = sr.logs_2pnormal(3.1, 0.5, 0.5, 4.0, backend=backend)
    assert np.isclose(res0, res)

    # multiple observations
    res = sr.logs_2pnormal(
        np.array([-2.9, 4.4]),
        np.array([1.1, 1.8]),
        np.array([10.1, 1.2]),
        backend=backend,
    )
    expected = np.array([6.116912, 8.046626])
    assert np.allclose(res, expected)


def test_logs_poisson(backend):
    ## test shape

    # one observation, multiple forecasts
    res = sr.logs_poisson(
        17,
        np.random.uniform(0, 20, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.logs_poisson(
        np.random.uniform(-5, 10, (4, 3)),
        3.2,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.logs_poisson(
        np.random.uniform(-5, 10, (4, 3)),
        np.random.uniform(0, 20, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # negative mean parameter
    with pytest.raises(ValueError):
        sr.logs_poisson(7, -0.1, backend=backend, check_pars=True)
        return

    # zero mean parameter
    with pytest.raises(ValueError):
        sr.logs_poisson(7, 0.0, backend=backend, check_pars=True)
        return

    ## test correctness

    # single observation
    res = sr.logs_poisson(1.0, 3.0, backend=backend)
    expected = 1.901388
    assert np.isclose(res, expected)

    res = sr.logs_poisson(1.5, 2.3, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    # negative observation
    res = sr.logs_poisson(-1.0, 1.5, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    # multiple observations
    res = sr.logs_poisson(np.array([2.0, 4.0]), np.array([10.1, 1.2]), backend=backend)
    expected = np.array([6.168076, 3.648768])
    assert np.allclose(res, expected)


def test_logs_t(backend):
    if backend == "torch":
        pytest.skip("Not implemented in torch backend")

    ## test shape

    # one observation, multiple forecasts
    res = sr.logs_t(
        17,
        np.random.uniform(0, 20, (4, 3)),
        np.random.uniform(10, 20, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.logs_t(
        np.random.uniform(-5, 20, (4, 3)),
        12,
        12,
        12,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.logs_t(
        np.random.uniform(-5, 20, (4, 3)),
        np.random.uniform(0, 20, (4, 3)),
        np.random.uniform(10, 20, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # negative df parameter
    with pytest.raises(ValueError):
        sr.logs_t(7, -4, backend=backend, check_pars=True)
        return

    # zero df parameter
    with pytest.raises(ValueError):
        sr.logs_t(7, 0, backend=backend, check_pars=True)
        return

    # negative scale parameter
    with pytest.raises(ValueError):
        sr.logs_t(7, 4, scale=-0.1, backend=backend, check_pars=True)
        return

    # zero scale parameter
    with pytest.raises(ValueError):
        sr.logs_t(7, 4, scale=0.0, backend=backend, check_pars=True)
        return

    ## test correctness

    # default location
    res0 = sr.logs_t(-3, 2.1, backend=backend)
    res = sr.logs_t(-3, 2.1, location=0, backend=backend)
    assert np.isclose(res0, res)

    # default scale
    res = sr.logs_t(-3, 2.1, location=0, scale=1.0, backend=backend)
    assert np.isclose(res0, res)

    # invariance to location shifts
    res = sr.logs_t(-3 + 0.1, 2.1, location=0.1, scale=1.0, backend=backend)
    assert np.isclose(res0, res)

    # single observation
    res = sr.logs_t(7.1, 4.2, 3.8, 0.3, backend=backend)
    expected = 8.600513
    assert np.isclose(res, expected)

    # negative observation
    res = sr.logs_t(-3.1, 2.9, 4.0, 0.5, backend=backend)
    expected = 8.60978
    assert np.isclose(res, expected)

    # multiple observations
    res = sr.logs_t(
        np.array([-2.9, 4.4]),
        np.array([5.1, 19.4]),
        np.array([1.1, 1.8]),
        np.array([10.1, 1.2]),
        backend=backend,
    )
    expected = np.array([3.372580, 3.324565])
    assert np.allclose(res, expected)


def test_logs_uniform(backend):
    ## test shape

    # one observation, multiple forecasts
    res = sr.logs_uniform(
        7,
        np.random.uniform(0, 5, (4, 3)),
        np.random.uniform(5, 10, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.logs_uniform(
        np.random.uniform(0, 10, (4, 3)),
        1,
        7,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.logs_uniform(
        np.random.uniform(0, 10, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        np.random.uniform(5, 10, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # min >= max
    with pytest.raises(ValueError):
        sr.logs_uniform(0.7, min=2, max=1, backend=backend, check_pars=True)
        return

    ## test correctness

    # default min
    res0 = sr.logs_uniform(0.7, backend=backend)
    res = sr.logs_uniform(0.7, min=0, backend=backend)
    assert np.isclose(res0, res)

    # default sigma
    res = sr.logs_uniform(0.7, min=0, max=1.0, backend=backend)
    assert np.isclose(res0, res)

    # invariance to location shifts
    res = sr.logs_uniform(0.7 + 0.1, min=0.1, max=1.1, backend=backend)
    assert np.isclose(res0, res)

    # single observation
    res = sr.logs_uniform(0.3, -1.0, 2.1, backend=backend)
    expected = 1.131402
    assert np.isclose(res, expected)

    res = sr.logs_uniform(2.2, 0.1, 3.1, backend=backend)
    expected = 1.098612
    assert np.isclose(res, expected)

    # observation outside of interval range
    res = sr.logs_uniform(-17.9, -15.2, -8.7, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    # multiple observations
    res = sr.logs_uniform(
        np.array([2.9, 4.4]),
        np.array([1.1, 1.8]),
        np.array([10.1, 5.1]),
        backend=backend,
    )
    expected = np.array([2.197225, 1.193922])
    assert np.allclose(res, expected)
