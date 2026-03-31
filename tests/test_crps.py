import numpy as np
import pytest
import scipy.stats as st
import scoringrules as sr

ENSEMBLE_SIZE = 11
N = 20

ESTIMATORS = ["nrg", "fair", "pwm", "int", "qd", "akr", "akr_circperm"]


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_crps_ensemble(estimator, backend):
    obs = np.random.randn(N)
    mu = obs + np.random.randn(N) * 0.3
    sigma = abs(np.random.randn(N)) * 0.5
    fct = np.random.randn(N, ENSEMBLE_SIZE) * sigma[..., None] + mu[..., None]

    # test exceptions
    with pytest.raises(ValueError):
        est = "undefined_estimator"
        sr.crps_ensemble(obs, fct, estimator=est, backend=backend)

    # test shape
    res = sr.crps_ensemble(obs, fct, estimator=estimator, backend=backend)
    assert res.shape == (N,)
    res = sr.crps_ensemble(
        obs,
        np.random.randn(ENSEMBLE_SIZE, N),
        m_axis=0,
        estimator=estimator,
        backend=backend,
    )
    assert res.shape == (N,)

    # non-negative values
    if estimator not in ["akr", "akr_circperm"]:
        res = sr.crps_ensemble(obs, fct, estimator=estimator, backend=backend)
        res = np.asarray(res)
        assert not np.any(res < 0.0)

    # test equivalence with and without weights
    w = np.ones(fct.shape)
    res_nrg_w = sr.crps_ensemble(
        obs, fct, ens_w=w, estimator=estimator, backend=backend
    )
    res_nrg_now = sr.crps_ensemble(obs, fct, estimator=estimator, backend=backend)
    assert np.allclose(res_nrg_w, res_nrg_now, atol=1e-6)

    # approx zero when perfect forecast
    perfect_fct = obs[..., None] + np.random.randn(N, ENSEMBLE_SIZE) * 0.00001
    res = sr.crps_ensemble(obs, perfect_fct, estimator=estimator, backend=backend)
    res = np.asarray(res)
    assert not np.any(res - 0.0 > 0.0001)


def test_crps_ensemble_corr(backend):
    obs = np.random.randn(N)
    mu = obs + np.random.randn(N) * 0.3
    sigma = abs(np.random.randn(N)) * 0.5
    fct = np.random.randn(N, ENSEMBLE_SIZE) * sigma[..., None] + mu[..., None]

    # test equivalence of different estimators
    res_nrg = sr.crps_ensemble(obs, fct, estimator="nrg", backend=backend)
    res_qd = sr.crps_ensemble(obs, fct, estimator="qd", backend=backend)
    res_fair = sr.crps_ensemble(obs, fct, estimator="fair", backend=backend)
    res_pwm = sr.crps_ensemble(obs, fct, estimator="pwm", backend=backend)
    if backend in ["torch", "jax"]:
        assert np.allclose(res_nrg, res_qd, rtol=1e-03)
        assert np.allclose(res_fair, res_pwm, rtol=1e-03)
    else:
        assert np.allclose(res_nrg, res_qd)
        assert np.allclose(res_fair, res_pwm)

    # test equivalence of different estimators with weights
    w = np.abs(np.random.randn(N, ENSEMBLE_SIZE) * sigma[..., None])
    res_nrg = sr.crps_ensemble(obs, fct, ens_w=w, estimator="nrg", backend=backend)
    res_qd = sr.crps_ensemble(obs, fct, ens_w=w, estimator="qd", backend=backend)
    if backend in ["torch", "jax"]:
        assert np.allclose(res_nrg, res_qd, rtol=1e-03)
    else:
        assert np.allclose(res_nrg, res_qd)

    # test correctness
    obs = -0.6042506
    fct = np.array(
        [
            1.7812118,
            0.5863797,
            0.7038174,
            -0.7743998,
            -0.2751647,
            1.1863249,
            1.2990966,
            -0.3242982,
            -0.5968781,
            0.9064937,
        ]
    )
    res = sr.crps_ensemble(obs, fct, estimator="qd")
    assert np.isclose(res, 0.6126602)

    w = np.arange(10)
    res = sr.crps_ensemble(obs, fct, ens_w=w, estimator="qd")
    assert np.isclose(res, 0.4923673)


def test_crps_quantile(backend):
    # test shape
    obs = np.random.randn(N)
    fct = np.random.randn(N, ENSEMBLE_SIZE)
    alpha = np.linspace(0.1, 0.9, ENSEMBLE_SIZE)
    res = sr.crps_quantile(obs, fct, alpha, backend=backend)
    assert res.shape == (N,)
    fct = np.random.randn(ENSEMBLE_SIZE, N)
    res = sr.crps_quantile(
        obs, np.random.randn(ENSEMBLE_SIZE, N), alpha, m_axis=0, backend=backend
    )
    assert res.shape == (N,)

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
            percentage_error_to_analytic = 1 - sr.crps_quantile(
                obs, fct, alpha, backend=backend
            ) / sr.crps_normal(obs, mu, 1, backend=backend)
            percentage_error_to_analytic = np.asarray(percentage_error_to_analytic)
            assert np.all(
                np.abs(percentage_error_to_analytic) < 1 / A
            ), "Quantile CRPS should be close to normal CRPS"

    # Test raise valueerror if array sizes don't match
    with pytest.raises(ValueError):
        sr.crps_quantile(obs, fct, alpha[0:42], backend=backend)
        return


def test_crps_beta(backend):
    if backend == "torch":
        pytest.skip("Not implemented in torch backend")

    ## test shape

    # one observation, multiple forecasts
    res = sr.crps_beta(
        0.1,
        np.random.uniform(0, 3, (4, 3)),
        np.random.uniform(0, 3, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.crps_beta(
        np.random.uniform(0, 1, (4, 3)),
        2.4,
        0.5,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.crps_beta(
        np.random.uniform(0, 1, (4, 3)),
        np.random.uniform(0, 3, (4, 3)),
        np.random.uniform(0, 3, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # lower bound smaller than upper bound
    with pytest.raises(ValueError):
        sr.crps_beta(
            0.3, 0.7, 1.1, lower=1.0, upper=0.0, backend=backend, check_pars=True
        )
        return

    # lower bound equal to upper bound
    with pytest.raises(ValueError):
        sr.crps_beta(
            0.3, 0.7, 1.1, lower=0.5, upper=0.5, backend=backend, check_pars=True
        )
        return

    # negative shape parameters
    with pytest.raises(ValueError):
        sr.crps_beta(0.3, -0.7, 1.1, backend=backend, check_pars=True)
        return

    with pytest.raises(ValueError):
        sr.crps_beta(0.3, 0.7, -1.1, backend=backend, check_pars=True)
        return

    ## correctness tests

    # single forecast
    res = sr.crps_beta(0.3, 0.7, 1.1, backend=backend)
    expected = 0.0850102437
    assert np.isclose(res, expected)

    # custom bounds
    res = sr.crps_beta(-3.0, 0.7, 1.1, lower=-5.0, upper=4.0, backend=backend)
    expected = 0.883206751
    assert np.isclose(res, expected)

    # observation outside bounds
    res = sr.crps_beta(-3.0, 0.7, 1.1, lower=-1.3, upper=4.0, backend=backend)
    expected = 2.880094
    assert np.isclose(res, expected)

    # multiple forecasts
    res = sr.crps_beta(
        -3.0,
        np.array([0.7, 2.0]),
        np.array([1.1, 4.8]),
        lower=np.array([-5.0, -5.0]),
        upper=np.array([4.0, 4.0]),
    )
    expected = np.array([0.8832068, 0.4149867])
    assert np.allclose(res, expected)


def test_crps_binomial(backend):
    if backend == "torch":
        pytest.skip("Not implemented in torch backend")

    ## test shape

    #    res = sr.crps_binomial(
    #        np.random.randint(0, 10, size=(4, 3)),
    #        np.full((4, 3), 10),
    #        np.random.uniform(0, 1, (4, 3)),
    #        backend=backend,
    #    )
    #    assert res.shape == (4, 3)
    #

    ones = np.ones(2)
    k, n, p = 8, 10, 0.9
    s = sr.crps_binomial(k * ones, n, p, backend=backend)
    assert np.isclose(s, np.array([0.6685115, 0.6685115])).all()
    s = sr.crps_binomial(k * ones, n * ones, p, backend=backend)
    assert np.isclose(s, np.array([0.6685115, 0.6685115])).all()
    s = sr.crps_binomial(k * ones, n * ones, p * ones, backend=backend)
    assert np.isclose(s, np.array([0.6685115, 0.6685115])).all()
    s = sr.crps_binomial(k, n * ones, p * ones, backend=backend)
    assert np.isclose(s, np.array([0.6685115, 0.6685115])).all()
    s = sr.crps_binomial(k * ones, n, p * ones, backend=backend)
    assert np.isclose(s, np.array([0.6685115, 0.6685115])).all()

    ## test exceptions

    # negative size parameter
    with pytest.raises(ValueError):
        sr.crps_binomial(7, -1, 0.9, backend=backend, check_pars=True)
        return

    # zero size parameter
    with pytest.raises(ValueError):
        sr.crps_binomial(2.1, 0, 0.1, backend=backend, check_pars=True)
        return

    # negative prob parameter
    with pytest.raises(ValueError):
        sr.crps_binomial(7, 15, -0.1, backend=backend, check_pars=True)
        return

    # prob parameter greater than one
    with pytest.raises(ValueError):
        sr.crps_binomial(1, 10, 1.1, backend=backend, check_pars=True)
        return

    # non-integer size parameter
    with pytest.raises(ValueError):
        sr.crps_binomial(4, 8.99, 0.5, backend=backend, check_pars=True)
        return

    ## test correctness

    # single observation
    res = sr.crps_binomial(8, 10, 0.9, backend=backend)
    expected = 0.6685115
    assert np.isclose(res, expected)

    # negative observation
    res = sr.crps_binomial(-8, 10, 0.9, backend=backend)
    expected = 16.49896
    assert np.isclose(res, expected)

    # zero prob parameter
    res = sr.crps_binomial(71, 212, 0, backend=backend, check_pars=True)
    expected = 71
    assert np.isclose(res, expected)

    # observation larger than size
    res = sr.crps_binomial(18, 10, 0.9, backend=backend)
    expected = 8.498957
    assert np.isclose(res, expected)

    # non-integer observation
    res = sr.crps_binomial(5.6, 10, 0.9, backend=backend)
    expected = 2.901231
    assert np.isclose(res, expected)

    # multiple observations
    res = sr.crps_binomial(np.array([5.6, 17.2]), 10, 0.9, backend=backend)
    expected = np.array([2.901231, 7.698957])
    assert np.allclose(res, expected)


def test_crps_exponential(backend):
    ## test shape

    # one observation, multiple forecasts
    res = sr.crps_exponential(
        7.2,
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.crps_exponential(
        np.random.uniform(0, 10, (4, 3)),
        7.2,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.crps_exponential(
        np.random.uniform(0, 10, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # negative rate
    with pytest.raises(ValueError):
        sr.crps_exponential(3.2, -1, backend=backend, check_pars=True)
        return

    # zero rate
    with pytest.raises(ValueError):
        sr.crps_exponential(3.2, 0, backend=backend, check_pars=True)
        return

    ## test correctness

    # single observation
    res = sr.crps_exponential(3, 0.7, backend=backend)
    expected = 1.20701837
    assert np.isclose(res, expected)

    # negative observation
    res = sr.crps_exponential(-3, 0.7, backend=backend)
    expected = 3.714286
    assert np.isclose(res, expected)

    # multiple observations
    res = sr.crps_exponential(np.array([5.6, 17.2]), 10.1, backend=backend)
    expected = np.array([5.451485, 17.051485])
    assert np.allclose(res, expected)


def test_crps_exponentialM(backend):
    ## test shape

    # one observation, multiple forecasts
    #    res = sr.crps_exponentialM(
    #        6.2,
    #        np.random.uniform(0, 5, (4, 3)),
    #        np.random.uniform(0, 5, (4, 3)),
    #        np.random.uniform(0, 1, (4, 3)),
    #        backend=backend,
    #    )
    #    assert res.shape == (4, 3)
    #

    # multiple observations, one forecast
    res = sr.crps_exponentialM(
        np.random.uniform(0, 10, (4, 3)),
        2.4,
        3.5,
        0.1,
        backend=backend,
    )

    # multiple observations, multiple forecasts
    res = sr.crps_exponentialM(
        np.random.uniform(0, 10, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        np.random.uniform(0, 1, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # negative scale
    with pytest.raises(ValueError):
        sr.crps_exponentialM(3.2, -1, -1, 0.0, backend=backend, check_pars=True)
        return

    # zero scale
    with pytest.raises(ValueError):
        sr.crps_exponentialM(3.2, -1, 0.0, 0.0, backend=backend, check_pars=True)
        return

    # negative mass
    with pytest.raises(ValueError):
        sr.crps_exponentialM(3.2, -1, 2, -0.1, backend=backend, check_pars=True)
        return

    # mass larger than one
    with pytest.raises(ValueError):
        sr.crps_exponentialM(3.2, -1, 2, 2.1, backend=backend, check_pars=True)
        return

    ## test correctness

    # single observation
    res = sr.crps_exponentialM(0.3, 0.0, 1.0, 0.1, backend=backend)
    expected = 0.2384728
    assert np.isclose(res, expected)

    # negative location
    res = sr.crps_exponentialM(0.3, -2.0, 3.0, 0.1, backend=backend)
    expected = 0.6236187
    assert np.isclose(res, expected)

    # negative observation
    res = sr.crps_exponentialM(-1.2, -2.0, 3.0, 0.1, backend=backend)
    expected = 0.751013
    assert np.isclose(res, expected)

    # observation smaller than location
    res = sr.crps_exponentialM(-1.2, 2.0, 3.0, 0.1, backend=backend)
    expected = 4.415
    assert np.isclose(res, expected)

    # zero mass
    res = sr.crps_exponentialM(-1.2, -2.0, 3.0, 0, backend=backend)
    expected = 0.89557
    assert np.isclose(res, expected)

    # multiple observations
    res = sr.crps_exponentialM(
        np.array([6.4, 2.2, 17.2]),
        10.1,
        4.1,
        np.array([0.1, 0.2, 0.3]),
        backend=backend,
    )
    expected = np.array([5.360500, 9.212000, 3.380377])
    assert np.allclose(res, expected)

    # check equivalence with crps_exponential
    res0 = sr.crps_exponential(8.2, 3, backend=backend)
    res = sr.crps_exponentialM(8.2, 0, 1 / 3, 0, backend=backend)
    assert np.isclose(res, res0)


def test_crps_2pexponential(backend):
    ## test shape

    # one observation, multiple forecasts
    res = sr.crps_2pexponential(
        6.2,
        np.random.uniform(0, 5, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        np.random.uniform(-2, 2, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.crps_2pexponential(
        np.random.uniform(-5, 5, (4, 3)),
        2.4,
        10.1,
        np.random.uniform(-2, 2, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.crps_2pexponential(
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
        sr.crps_2pexponential(3.2, -1, 2, 2.1, backend=backend, check_pars=True)
        return

    with pytest.raises(ValueError):
        sr.crps_2pexponential(3.2, 1, -2, 2.1, backend=backend, check_pars=True)
        return

    # zero scale parameters
    with pytest.raises(ValueError):
        sr.crps_2pexponential(3.2, 0, 2, 2.1, backend=backend, check_pars=True)
        return

    with pytest.raises(ValueError):
        sr.crps_2pexponential(3.2, 1, 0, 2.1, backend=backend, check_pars=True)
        return

    ## test correctness

    # single observation
    res = sr.crps_2pexponential(0.3, 0.1, 4.3, 0.0, backend=backend)
    expected = 1.787032
    assert np.isclose(res, expected)

    # negative location
    res = sr.crps_2pexponential(-20.8, 7.1, 2.0, -25.4, backend=backend)
    expected = 6.018359
    assert np.isclose(res, expected)

    # multiple observations
    res = sr.crps_2pexponential(
        np.array([-1.2, 3.7]), 0.1, 4.3, np.array([-0.1, 0.1]), backend=backend
    )
    expected = np.array([3.1488637, 0.8873341])
    assert np.allclose(res, expected)

    # check equivalence with laplace distribution
    res = sr.crps_2pexponential(-20.8, 2.0, 2.0, -25.4, backend=backend)
    res0 = sr.crps_laplace(-20.8, -25.4, 2.0, backend=backend)
    assert np.isclose(res, res0)


def test_crps_gamma(backend):
    ## test shape

    # one observation, multiple forecasts
    res = sr.crps_gamma(
        6.2,
        np.random.uniform(0, 5, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.crps_gamma(
        np.random.uniform(0, 10, (4, 3)),
        4.0,
        2.1,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.crps_gamma(
        np.random.uniform(0, 10, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # rate and scale provided
    with pytest.raises(ValueError):
        sr.crps_gamma(3.2, -1, rate=2, scale=0.4, backend=backend, check_pars=True)
        return

    # neither rate nor scale provided
    with pytest.raises(ValueError):
        sr.crps_gamma(3.2, -1, backend=backend, check_pars=True)
        return

    # negative shape parameter
    with pytest.raises(ValueError):
        sr.crps_gamma(3.2, -1, 2, backend=backend, check_pars=True)
        return

    # negative rate/scale parameter
    with pytest.raises(ValueError):
        sr.crps_gamma(3.2, 1, -2, backend=backend, check_pars=True)
        return

    # zero shape parameter
    with pytest.raises(ValueError):
        sr.crps_gamma(3.2, 0.0, 2, backend=backend, check_pars=True)
        return

    ## test correctness

    # single observation
    res = sr.crps_gamma(0.2, 1.1, 0.7, backend=backend)
    expected = 0.6343718
    assert np.isclose(res, expected)

    # scale instead of rate
    res = sr.crps_gamma(0.2, 1.1, scale=1 / 0.7, backend=backend)
    assert np.isclose(res, expected)

    # negative observation
    res = sr.crps_gamma(-4.2, 1.1, rate=0.7, backend=backend)
    expected = 5.014442
    assert np.isclose(res, expected)

    # multiple observations
    res = sr.crps_gamma(np.array([-1.2, 2.3]), 1.1, 0.7, backend=backend)
    expected = np.array([2.0144417, 0.6500017])
    assert np.allclose(res, expected)

    # check equivalence with exponential distribution
    res0 = sr.crps_exponential(3.1, 2, backend=backend)
    res = sr.crps_gamma(3.1, 1, 2, backend=backend)
    assert np.isclose(res0, res)


def test_crps_csg0(backend):
    ## test shape

    # one observation, multiple forecasts
    res = sr.crps_csg0(
        6.2,
        np.random.uniform(0, 5, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        shift=np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.crps_csg0(
        np.random.uniform(0, 5, (4, 3)),
        9.8,
        8.8,
        shift=1.1,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.crps_csg0(
        np.random.uniform(0, 5, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        shift=np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # rate and scale provided
    with pytest.raises(ValueError):
        sr.crps_csg0(0.7, 0.5, rate=2.0, scale=0.5, shift=0.3, backend=backend)
        return

    # neither rate nor scale provided
    with pytest.raises(ValueError):
        sr.crps_csg0(0.7, 0.5, shift=0.3, backend=backend)
        return

    # negative shape parameter
    with pytest.raises(ValueError):
        sr.crps_csg0(0.7, -0.5, 2.0, shift=2, backend=backend, check_pars=True)
        return

    # negative rate/scale parameter
    with pytest.raises(ValueError):
        sr.crps_csg0(0.7, 0.5, -2.0, shift=2, backend=backend, check_pars=True)
        return

    # zero shape parameter
    with pytest.raises(ValueError):
        sr.crps_csg0(0.7, 0, 2.0, shift=2, backend=backend, check_pars=True)
        return

    ## test correctness

    obs, shape, rate, shift = 0.7, 0.5, 2.0, 0.3
    expected = 0.5411044

    # single observation
    res = sr.crps_csg0(obs, shape=shape, rate=rate, shift=shift, backend=backend)
    assert np.isclose(res, expected)

    # scale instead of rate
    res = sr.crps_csg0(obs, shape=shape, scale=1.0 / rate, shift=shift, backend=backend)
    assert np.isclose(res, expected)

    # with and without shift parameter
    res0 = sr.crps_csg0(obs, shape, rate, backend=backend)
    res = sr.crps_csg0(obs, shape, rate, shift=0.0, backend=backend)
    assert np.isclose(res0, res)

    # check equivalence with gamma distribution
    res0 = sr.crps_gamma(obs, shape, rate, backend=backend)
    assert np.isclose(res0, res)


def test_crps_gev(backend):
    if backend == "torch":
        pytest.skip("Not implemented in torch backend")

    ## test shape

    # one observation, multiple forecasts
    res = sr.crps_gev(
        6.2,
        np.random.uniform(-10, 1, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        0.5,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.crps_gev(
        np.random.uniform(-5, 5, (4, 3)),
        -4.9,
        2.3,
        0.7,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.crps_gev(
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
        sr.crps_gev(0.7, 0.5, 2.0, -0.5, backend=backend, check_pars=True)
        return

    # zero scale parameter
    with pytest.raises(ValueError):
        sr.crps_gev(0.7, 0.5, 2.0, 0.0, backend=backend, check_pars=True)
        return

    # shape parameter greater than one
    with pytest.raises(ValueError):
        sr.crps_gev(0.7, 1.5, 0.0, 0.5, backend=backend, check_pars=True)
        return

    ## test correctness

    # test default location
    res0 = sr.crps_gev(0.3, 0.0, backend=backend)
    res = sr.crps_gev(0.3, 0.0, 0.0, backend=backend)
    assert np.isclose(res0, res)

    # test default scale
    res = sr.crps_gev(0.3, 0.0, 0.0, 1.0, backend=backend)
    assert np.isclose(res0, res)

    # test invariance to shift
    mu = 0.1
    res = sr.crps_gev(0.3 + mu, 0.0, mu, 1.0, backend=backend)
    assert np.isclose(res0, res)

    # test invariance to rescaling
    sigma = 0.9
    res = sr.crps_gev(0.3 * sigma, 0.0, scale=sigma, backend=backend)
    assert np.isclose(res0 * sigma, res)

    # positive shape parameter
    res = sr.crps_gev(0.3, 0.7, backend=backend)
    expected = 0.458044365
    assert np.isclose(res, expected)

    # negative shape parameter
    res = sr.crps_gev(0.3, -0.7, backend=backend)
    expected = 0.207621488
    assert np.isclose(res, expected)

    # zero shape parameter
    res = sr.crps_gev(0.3, 0.0, backend=backend)
    expected = 0.276441
    assert np.isclose(res, expected)

    # multiple observations
    res = sr.crps_gev(np.array([0.9, -2.1]), -0.7, 0, 1.4, backend=backend)
    expected = np.array([0.3575414, 1.7008711])
    assert np.allclose(res, expected)


def test_crps_gpd(backend):
    ## test shape

    # one observation, multiple forecasts
    res = sr.crps_gpd(
        6.2,
        np.random.uniform(-10, 1, (4, 3)),
        np.random.uniform(-5, 5, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.crps_gpd(
        np.random.uniform(-5, 5, (4, 3)),
        -4.9,
        2.3,
        0.7,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.crps_gpd(
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
        sr.crps_gpd(0.7, 0.5, 2.0, -0.5, backend=backend, check_pars=True)
        return

    # zero scale parameter
    with pytest.raises(ValueError):
        sr.crps_gpd(0.7, 0.5, 2.0, 0.0, backend=backend, check_pars=True)
        return

    # shape parameter greater than one
    with pytest.raises(ValueError):
        sr.crps_gpd(0.7, 1.5, 0.0, 0.5, backend=backend, check_pars=True)
        return

    # mass parameter smaller than zero
    with pytest.raises(ValueError):
        sr.crps_gpd(0.7, 0.5, 0.0, 0.5, mass=-0.1, backend=backend, check_pars=True)
        return

    # mass parameter smaller than zero
    with pytest.raises(ValueError):
        sr.crps_gpd(0.7, 0.5, 0.0, 0.5, mass=-0.1, backend=backend, check_pars=True)
        return

    # mass parameter greater than one
    with pytest.raises(ValueError):
        sr.crps_gpd(0.7, 0.5, 0.0, 0.5, mass=1.1, backend=backend, check_pars=True)
        return

    ## test correctness

    # test default location
    res0 = sr.crps_gpd(0.3, 0.0, backend=backend)
    res = sr.crps_gpd(0.3, 0.0, 0.0, backend=backend)
    assert np.isclose(res0, res)

    # test default scale
    res = sr.crps_gpd(0.3, 0.0, 0.0, 1.0, backend=backend)
    assert np.isclose(res0, res)

    # test default mass
    res = sr.crps_gpd(0.3, 0.0, 0.0, 1.0, 0.0, backend=backend)
    assert np.isclose(res0, res)

    # test invariance to shift
    res = sr.crps_gpd(0.3 + 0.1, 0.0, 0.1, 1.0, backend=backend)
    assert np.isclose(res0, res)

    # test invariance to rescaling
    res0 = sr.crps_gpd((0.3 - 0.1) / 0.8, 0.4, backend=backend)
    res = sr.crps_gpd(0.3, 0.4, 0.1, 0.8, backend=backend)
    assert np.isclose(res0 * 0.8, res)

    # single observation
    res = sr.crps_gpd(0.3, 0.9, backend=backend)
    expected = 0.6849332
    assert np.isclose(res, expected)

    # negative observation
    res = sr.crps_gpd(-0.3, 0.9, backend=backend)
    expected = 1.209091
    assert np.isclose(res, expected)

    # negative shape parameter
    res = sr.crps_gpd(0.3, -0.9, backend=backend)
    expected = 0.1338672
    assert np.isclose(res, expected)

    # non-zero mass parameter
    res = sr.crps_gpd(-0.3, -0.9, -2.1, 10.0, mass=0.3, backend=backend)
    expected = 1.195042
    assert np.isclose(res, expected)

    # mass parameter equal to one
    res = sr.crps_gpd(-0.3, -0.9, -2.1, 10.0, mass=1.0, backend=backend)
    expected = 1.8
    assert np.isclose(res, expected)

    # multiple observations
    res = sr.crps_gpd(np.array([0.9, -2.1]), -0.7, 0, 1.4, backend=backend)
    expected = np.array([0.1570813, 2.6185185])
    assert np.allclose(res, expected)


def test_crps_gtclogis(backend):
    ## test shape

    # one observation, multiple forecasts
    res = sr.crps_gtclogistic(
        17,
        np.random.uniform(10, 20, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.crps_gtclogistic(
        np.random.uniform(-10, 10, (4, 3)),
        3.2,
        4.1,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.crps_gtclogistic(
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # negative scale parameter
    with pytest.raises(ValueError):
        sr.crps_gtclogistic(0.7, 2.0, -0.5, backend=backend, check_pars=True)
        return

    # zero scale parameter
    with pytest.raises(ValueError):
        sr.crps_gtclogistic(0.7, 2.0, 0.0, backend=backend, check_pars=True)
        return

    # negative lower mass parameter
    with pytest.raises(ValueError):
        sr.crps_gtclogistic(
            0.7, 2.0, 0.4, lower=-1, lmass=-0.4, backend=backend, check_pars=True
        )
        return

    # lower mass parameter larger than one
    with pytest.raises(ValueError):
        sr.crps_gtclogistic(
            0.7, 2.0, 0.4, lower=-1, lmass=1.4, backend=backend, check_pars=True
        )
        return

    # negative upper mass parameter
    with pytest.raises(ValueError):
        sr.crps_gtclogistic(
            0.7, 2.0, 0.4, upper=4, umass=-0.4, backend=backend, check_pars=True
        )
        return

    # upper mass parameter larger than one
    with pytest.raises(ValueError):
        sr.crps_gtclogistic(
            0.7, 2.0, 0.4, upper=4, umass=1.4, backend=backend, check_pars=True
        )
        return

    # sum of lower and upper mass parameters larger than one
    with pytest.raises(ValueError):
        sr.crps_gtclogistic(
            0.7,
            2.0,
            0.4,
            lower=-1,
            upper=4,
            lmass=0.7,
            umass=0.4,
            backend=backend,
            check_pars=True,
        )
        return

    # lower bound larger than upper bound
    with pytest.raises(ValueError):
        sr.crps_gtclogistic(
            0.7, 2.0, 0.4, lower=-1, upper=-4, backend=backend, check_pars=True
        )
        return

    ## test correctness

    # default values
    res0 = sr.crps_gtclogistic(1.8, backend=backend)
    res = sr.crps_gtclogistic(1.8, 0.0, 1.0, -np.inf, np.inf, 0.0, 0.0, backend=backend)
    assert np.isclose(res0, res)

    # single observation
    res = sr.crps_gtclogistic(1.8, -3.0, 3.3, -5.0, 4.7, 0.1, 0.15, backend=backend)
    expected = 1.599721
    assert np.isclose(res, expected)

    # observation below lower bound
    res = sr.crps_gtclogistic(-5.8, -3.0, 3.3, -5.0, 4.7, 0.1, 0.15, backend=backend)
    expected = 3.393796
    assert np.isclose(res, expected)

    # observation above upper bound
    res = sr.crps_gtclogistic(7.8, -3.0, 3.3, -5.0, 4.7, 0.1, 0.15, backend=backend)
    expected = 6.378633
    assert np.isclose(res, expected)

    # mass on lower bound equal to one
    res = sr.crps_gtclogistic(7.8, -3.0, 3.3, -5.0, 4.7, 1.0, 0.0, backend=backend)
    expected = 12.8
    assert np.isclose(res, expected)

    # mass on upper bound equal to one
    res = sr.crps_gtclogistic(7.8, -3.0, 3.3, -5.0, 4.7, 0.0, 1.0, backend=backend)
    expected = 3.1
    assert np.isclose(res, expected)

    # lower bound equal to infinite
    res = sr.crps_gtclogistic(
        7.8, -3.0, 3.3, float("-inf"), 4.7, 0.0, 0.15, backend=backend
    )
    expected = 7.444611
    assert np.isclose(res, expected)

    # upper bound equal to infinite
    res = sr.crps_gtclogistic(
        7.8, -3.0, 3.3, -5.0, float("inf"), 0.15, 0.0, backend=backend
    )
    expected = 6.339197
    assert np.isclose(res, expected)

    # aligns with crps_logistic
    res0 = sr.crps_logistic(1.8, -3.0, 3.3, backend=backend)
    res = sr.crps_gtclogistic(1.8, -3.0, 3.3, backend=backend)
    assert np.isclose(res, res0)

    # aligns with crps_tlogistic
    res0 = sr.crps_tlogistic(1.8, -3.0, 3.3, -5.0, 4.7, backend=backend)
    res = sr.crps_gtclogistic(1.8, -3.0, 3.3, -5.0, 4.7, backend=backend)
    assert np.isclose(res, res0)

    # multiple observations
    res = sr.crps_gtclogistic(
        np.array([1.8, -2.3]), 9.1, 11.1, -4.0, np.inf, 0.5, 0.0, backend=backend
    )
    expected = np.array([3.627106, 3.270710])
    assert np.allclose(res, expected)


def test_crps_tlogis(backend):
    ## test shape

    # one observation, multiple forecasts
    res = sr.crps_tlogistic(
        17,
        np.random.uniform(10, 20, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.crps_tlogistic(
        np.random.uniform(-10, 10, (4, 3)),
        3.2,
        4.1,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.crps_tlogistic(
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # negative scale parameter
    with pytest.raises(ValueError):
        sr.crps_tlogistic(0.7, 2.0, -0.5, backend=backend, check_pars=True)
        return

    # zero scale parameter
    with pytest.raises(ValueError):
        sr.crps_tlogistic(0.7, 2.0, 0.0, backend=backend, check_pars=True)
        return

    # lower bound larger than upper bound
    with pytest.raises(ValueError):
        sr.crps_tlogistic(
            0.7, 2.0, 0.4, lower=-1, upper=-4, backend=backend, check_pars=True
        )
        return

    ## test correctness

    # default values
    res0 = sr.crps_tlogistic(1.8, backend=backend)
    res = sr.crps_tlogistic(1.8, 0.0, 1.0, -np.inf, np.inf, backend=backend)
    assert np.isclose(res0, res)

    # single observation
    res = sr.crps_tlogistic(1.8, -3.0, 3.3, -5.0, 4.7, backend=backend)
    expected = 1.72305
    assert np.isclose(res, expected)

    # observation below lower bound
    res = sr.crps_tlogistic(-5.8, -3.0, 3.3, -5.0, 4.7, backend=backend)
    expected = 3.395149
    assert np.isclose(res, expected)

    # observation above upper bound
    res = sr.crps_tlogistic(7.8, -3.0, 3.3, -5.0, 4.7, backend=backend)
    expected = 7.254933
    assert np.isclose(res, expected)

    # aligns with crps_logistic
    res0 = sr.crps_logistic(1.8, -3.0, 3.3, backend=backend)
    res = sr.crps_tlogistic(1.8, -3.0, 3.3, backend=backend)
    assert np.isclose(res, res0)

    # multiple observations
    res = sr.crps_tlogistic(
        np.array([1.8, -2.3]), 9.1, 11.1, -4.0, np.inf, backend=backend
    )
    expected = np.array([7.932799, 11.320006])
    assert np.allclose(res, expected)


def test_crps_clogis(backend):
    ## test shape

    # one observation, multiple forecasts
    res = sr.crps_clogistic(
        17,
        np.random.uniform(10, 20, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.crps_clogistic(
        np.random.uniform(-10, 10, (4, 3)),
        3.2,
        4.1,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.crps_clogistic(
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # negative scale parameter
    with pytest.raises(ValueError):
        sr.crps_clogistic(0.7, 2.0, -0.5, backend=backend, check_pars=True)
        return

    # zero scale parameter
    with pytest.raises(ValueError):
        sr.crps_clogistic(0.7, 2.0, 0.0, backend=backend, check_pars=True)
        return

    # lower bound larger than upper bound
    with pytest.raises(ValueError):
        sr.crps_clogistic(
            0.7, 2.0, 0.4, lower=-1, upper=-4, backend=backend, check_pars=True
        )
        return

    ## test correctness

    # default values
    res0 = sr.crps_clogistic(1.8, backend=backend)
    res = sr.crps_clogistic(1.8, 0.0, 1.0, -np.inf, np.inf, backend=backend)
    assert np.isclose(res0, res)

    # single observation
    res = sr.crps_clogistic(1.8, -3.0, 3.3, -5.0, 4.7, backend=backend)
    expected = 2.599499
    assert np.isclose(res, expected)

    # observation below lower bound
    res = sr.crps_clogistic(-5.8, -3.0, 3.3, -5.0, 4.7, backend=backend)
    expected = 2.087692
    assert np.isclose(res, expected)

    # observation above upper bound
    res = sr.crps_clogistic(7.8, -3.0, 3.3, -5.0, 4.7, backend=backend)
    expected = 7.825271
    assert np.isclose(res, expected)

    # aligns with crps_logistic
    res0 = sr.crps_logistic(1.8, -3.0, 3.3, backend=backend)
    res = sr.crps_clogistic(1.8, -3.0, 3.3, backend=backend)
    assert np.isclose(res, res0)

    # multiple observations
    res = sr.crps_clogistic(
        np.array([1.8, -2.3]), 9.1, 11.1, -4.0, np.inf, backend=backend
    )
    expected = np.array([5.102037, 6.729603])
    assert np.allclose(res, expected)


def test_crps_gtcnormal(backend):
    ## test shape

    # one observation, multiple forecasts
    res = sr.crps_gtcnormal(
        17,
        np.random.uniform(10, 20, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.crps_gtcnormal(
        np.random.uniform(-10, 10, (4, 3)),
        3.2,
        4.1,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.crps_gtcnormal(
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # negative scale parameter
    with pytest.raises(ValueError):
        sr.crps_gtcnormal(0.7, 2.0, -0.5, backend=backend, check_pars=True)
        return

    # zero scale parameter
    with pytest.raises(ValueError):
        sr.crps_gtcnormal(0.7, 2.0, 0.0, backend=backend, check_pars=True)
        return

    # negative lower mass parameter
    with pytest.raises(ValueError):
        sr.crps_gtcnormal(
            0.7, 2.0, 0.4, lower=-1, lmass=-0.4, backend=backend, check_pars=True
        )
        return

    # lower mass parameter larger than one
    with pytest.raises(ValueError):
        sr.crps_gtcnormal(
            0.7, 2.0, 0.4, lower=-1, lmass=1.4, backend=backend, check_pars=True
        )
        return

    # negative upper mass parameter
    with pytest.raises(ValueError):
        sr.crps_gtcnormal(
            0.7, 2.0, 0.4, upper=4, umass=-0.4, backend=backend, check_pars=True
        )
        return

    # upper mass parameter larger than one
    with pytest.raises(ValueError):
        sr.crps_gtcnormal(
            0.7, 2.0, 0.4, upper=4, umass=1.4, backend=backend, check_pars=True
        )
        return

    # sum of lower and upper mass parameters larger than one
    with pytest.raises(ValueError):
        sr.crps_gtcnormal(
            0.7,
            2.0,
            0.4,
            lower=-1,
            upper=4,
            lmass=0.7,
            umass=0.4,
            backend=backend,
            check_pars=True,
        )
        return

    # lower bound larger than upper bound
    with pytest.raises(ValueError):
        sr.crps_gtcnormal(
            0.7, 2.0, 0.4, lower=-1, upper=-4, backend=backend, check_pars=True
        )
        return

    ## test correctness

    # default values
    res0 = sr.crps_gtcnormal(1.8, backend=backend)
    res = sr.crps_gtcnormal(1.8, 0.0, 1.0, -np.inf, np.inf, 0.0, 0.0, backend=backend)
    assert np.isclose(res0, res)

    # single observation
    res = sr.crps_gtcnormal(1.8, -3.0, 3.3, -5.0, 4.7, 0.1, 0.15, backend=backend)
    expected = 1.986411
    assert np.isclose(res, expected)

    # observation below lower bound
    res = sr.crps_gtcnormal(-5.8, -3.0, 3.3, -5.0, 4.7, 0.1, 0.15, backend=backend)
    expected = 2.993124
    assert np.isclose(res, expected)

    # observation above upper bound
    res = sr.crps_gtcnormal(7.8, -3.0, 3.3, -5.0, 4.7, 0.1, 0.15, backend=backend)
    expected = 6.974827
    assert np.isclose(res, expected)

    # mass on lower bound equal to one
    res = sr.crps_gtcnormal(7.8, -3.0, 3.3, -5.0, 4.7, 1.0, 0.0, backend=backend)
    expected = 12.8
    assert np.isclose(res, expected)

    # mass on upper bound equal to one
    res = sr.crps_gtcnormal(7.8, -3.0, 3.3, -5.0, 4.7, 0.0, 1.0, backend=backend)
    expected = 3.1
    assert np.isclose(res, expected)

    # aligns with crps_normal
    res0 = sr.crps_normal(1.8, -3.0, 3.3, backend=backend)
    res = sr.crps_gtcnormal(1.8, -3.0, 3.3, backend=backend)
    assert np.isclose(res, res0)

    # aligns with crps_tnormal
    res0 = sr.crps_tnormal(1.8, -3.0, 3.3, -5.0, 4.7, backend=backend)
    res = sr.crps_gtcnormal(1.8, -3.0, 3.3, -5.0, 4.7, backend=backend)
    assert np.isclose(res, res0)

    # multiple observations
    res = sr.crps_gtcnormal(
        np.array([1.8, -2.3]), 9.1, 11.1, -4.0, np.inf, 0.5, 0.0, backend=backend
    )
    expected = np.array([3.019942, 2.637561])
    assert np.allclose(res, expected)


def test_crps_tnormal(backend):
    ## test shape

    # one observation, multiple forecasts
    res = sr.crps_tnormal(
        17,
        np.random.uniform(10, 20, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.crps_tnormal(
        np.random.uniform(-10, 10, (4, 3)),
        3.2,
        4.1,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.crps_tnormal(
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # negative scale parameter
    with pytest.raises(ValueError):
        sr.crps_tnormal(0.7, 2.0, -0.5, backend=backend, check_pars=True)
        return

    # zero scale parameter
    with pytest.raises(ValueError):
        sr.crps_tnormal(0.7, 2.0, 0.0, backend=backend, check_pars=True)
        return

    # lower bound larger than upper bound
    with pytest.raises(ValueError):
        sr.crps_tnormal(
            0.7, 2.0, 0.4, lower=-1, upper=-4, backend=backend, check_pars=True
        )
        return

    ## test correctness

    # default values
    res0 = sr.crps_tnormal(1.8, backend=backend)
    res = sr.crps_tnormal(1.8, 0.0, 1.0, -np.inf, np.inf, backend=backend)
    assert np.isclose(res0, res)

    # single observation
    res = sr.crps_tnormal(1.8, -3.0, 3.3, -5.0, 4.7, backend=backend)
    expected = 2.326391
    assert np.isclose(res, expected)

    # observation below lower bound
    res = sr.crps_tnormal(-5.8, -3.0, 3.3, -5.0, 4.7, backend=backend)
    expected = 2.948674
    assert np.isclose(res, expected)

    # observation above upper bound
    res = sr.crps_tnormal(7.8, -3.0, 3.3, -5.0, 4.7, backend=backend)
    expected = 8.137612
    assert np.isclose(res, expected)

    # aligns with crps_normal
    res0 = sr.crps_normal(1.8, -3.0, 3.3, backend=backend)
    res = sr.crps_tnormal(1.8, -3.0, 3.3, backend=backend)
    assert np.isclose(res, res0)

    # multiple observations
    res = sr.crps_tnormal(
        np.array([1.8, -2.3]), 9.1, 11.1, -4.0, np.inf, backend=backend
    )
    expected = np.array([5.452673, 8.787913])
    assert np.allclose(res, expected)


def test_crps_cnormal(backend):
    ## test shape

    # one observation, multiple forecasts
    res = sr.crps_cnormal(
        17,
        np.random.uniform(10, 20, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.crps_cnormal(
        np.random.uniform(-10, 10, (4, 3)),
        3.2,
        4.1,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.crps_cnormal(
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # negative scale parameter
    with pytest.raises(ValueError):
        sr.crps_cnormal(0.7, 2.0, -0.5, backend=backend, check_pars=True)
        return

    # zero scale parameter
    with pytest.raises(ValueError):
        sr.crps_cnormal(0.7, 2.0, 0.0, backend=backend, check_pars=True)
        return

    # lower bound larger than upper bound
    with pytest.raises(ValueError):
        sr.crps_cnormal(
            0.7, 2.0, 0.4, lower=-1, upper=-4, backend=backend, check_pars=True
        )
        return

    ## test correctness

    # default values
    res0 = sr.crps_cnormal(1.8, backend=backend)
    res = sr.crps_cnormal(1.8, 0.0, 1.0, -np.inf, np.inf, backend=backend)
    assert np.isclose(res0, res)

    # single observation
    res = sr.crps_cnormal(1.8, -3.0, 3.3, -5.0, 4.7, backend=backend)
    expected = 3.068522
    assert np.isclose(res, expected)

    # observation below lower bound
    res = sr.crps_cnormal(-5.8, -3.0, 3.3, -5.0, 4.7, backend=backend)
    expected = 1.956462
    assert np.isclose(res, expected)

    # observation above upper bound
    res = sr.crps_cnormal(7.8, -3.0, 3.3, -5.0, 4.7, backend=backend)
    expected = 8.87606
    assert np.isclose(res, expected)

    # aligns with crps_normal
    res0 = sr.crps_cnormal(1.8, -3.0, 3.3, backend=backend)
    res = sr.crps_cnormal(1.8, -3.0, 3.3, backend=backend)
    assert np.isclose(res, res0)

    # multiple observations
    res = sr.crps_cnormal(
        np.array([1.8, -2.3]), 9.1, 11.1, -4.0, np.inf, backend=backend
    )
    expected = np.array([4.401269, 6.851981])
    assert np.allclose(res, expected)


def test_crps_gtct(backend):
    if backend in ["jax", "torch", "tensorflow"]:
        pytest.skip("Not implemented in jax, torch or tensorflow backends")

    ## test shape

    # one observation, multiple forecasts
    res = sr.crps_gtct(
        17,
        np.random.uniform(0, 20, (4, 3)),
        np.random.uniform(10, 20, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.crps_gtct(
        np.random.uniform(-10, 10, (4, 3)),
        11.1,
        3.2,
        4.1,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.crps_gtct(
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
        sr.crps_gtct(0.7, -1.7, 2.0, 0.5, backend=backend, check_pars=True)
        return

    # zero df parameter
    with pytest.raises(ValueError):
        sr.crps_gtct(0.7, 0.0, 2.0, 0.5, backend=backend, check_pars=True)
        return

    # negative scale parameter
    with pytest.raises(ValueError):
        sr.crps_gtct(0.7, 4.7, 2.0, -0.5, backend=backend, check_pars=True)
        return

    # zero scale parameter
    with pytest.raises(ValueError):
        sr.crps_gtct(0.7, 4.7, 2.0, 0.0, backend=backend, check_pars=True)
        return

    # negative lower mass parameter
    with pytest.raises(ValueError):
        sr.crps_gtct(
            0.7, 4.7, 2.0, 0.4, lower=-1, lmass=-0.4, backend=backend, check_pars=True
        )
        return

    # lower mass parameter larger than one
    with pytest.raises(ValueError):
        sr.crps_gtct(
            0.7, 4.7, 2.0, 0.4, lower=-1, lmass=1.4, backend=backend, check_pars=True
        )
        return

    # negative upper mass parameter
    with pytest.raises(ValueError):
        sr.crps_gtct(
            0.7, 4.7, 2.0, 0.4, upper=4, umass=-0.4, backend=backend, check_pars=True
        )
        return

    # upper mass parameter larger than one
    with pytest.raises(ValueError):
        sr.crps_gtct(
            0.7, 4.7, 2.0, 0.4, upper=4, umass=1.4, backend=backend, check_pars=True
        )
        return

    # sum of lower and upper mass parameters larger than one
    with pytest.raises(ValueError):
        sr.crps_gtct(
            0.7,
            4.7,
            2.0,
            0.4,
            lower=-1,
            upper=4,
            lmass=0.7,
            umass=0.4,
            backend=backend,
            check_pars=True,
        )
        return

    # lower bound larger than upper bound
    with pytest.raises(ValueError):
        sr.crps_gtct(
            0.7, 4.7, 2.0, 0.4, lower=-1, upper=-4, backend=backend, check_pars=True
        )
        return

    ## test correctness

    # default values
    res0 = sr.crps_gtct(1.8, 6.9, backend=backend)
    res = sr.crps_gtct(1.8, 6.9, 0.0, 1.0, -np.inf, np.inf, 0.0, 0.0, backend=backend)
    assert np.isclose(res0, res)

    # single observation
    res = sr.crps_gtct(1.8, 6.9, -3.0, 3.3, -5.0, 4.7, 0.1, 0.15, backend=backend)
    expected = 1.963135
    assert np.isclose(res, expected)

    # observation below lower bound
    res = sr.crps_gtct(-5.8, 6.9, -3.0, 3.3, -5.0, 4.7, 0.1, 0.15, backend=backend)
    expected = 3.016045
    assert np.isclose(res, expected)

    # observation above upper bound
    res = sr.crps_gtct(7.8, 6.9, -3.0, 3.3, -5.0, 4.7, 0.1, 0.15, backend=backend)
    expected = 6.921271
    assert np.isclose(res, expected)

    # mass on lower bound equal to one
    res = sr.crps_gtct(7.8, 6.9, -3.0, 3.3, -5.0, 4.7, 1.0, 0.0, backend=backend)
    expected = 12.8
    assert np.isclose(res, expected)

    # mass on upper bound equal to one
    res = sr.crps_gtct(7.8, 6.9, -3.0, 3.3, -5.0, 4.7, 0.0, 1.0, backend=backend)
    expected = 3.1
    assert np.isclose(res, expected)

    # aligns with crps_t
    res0 = sr.crps_t(1.8, 6.9, -3.0, 3.3, backend=backend)
    res = sr.crps_gtct(1.8, 6.9, -3.0, 3.3, backend=backend)
    assert np.isclose(res, res0)

    # aligns with crps_tt
    res0 = sr.crps_tt(1.8, 6.9, -3.0, 3.3, -5.0, 4.7, backend=backend)
    res = sr.crps_gtct(1.8, 6.9, -3.0, 3.3, -5.0, 4.7, backend=backend)
    assert np.isclose(res, res0)

    # multiple observations
    res = sr.crps_gtct(
        np.array([1.8, -2.3]), 6.9, 9.1, 11.1, -4.0, np.inf, 0.5, 0.0, backend=backend
    )
    expected = np.array([3.085881, 2.720847])
    assert np.allclose(res, expected)


def test_crps_tt(backend):
    if backend in ["jax", "torch", "tensorflow"]:
        pytest.skip("Not implemented in jax, torch or tensorflow backends")

    ## test shape

    # one observation, multiple forecasts
    res = sr.crps_tt(
        17,
        np.random.uniform(0, 20, (4, 3)),
        np.random.uniform(10, 20, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.crps_tt(
        np.random.uniform(-10, 10, (4, 3)),
        11.1,
        3.2,
        4.1,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.crps_tt(
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
        sr.crps_tt(0.7, -1.7, 2.0, 0.5, backend=backend, check_pars=True)
        return

    # zero df parameter
    with pytest.raises(ValueError):
        sr.crps_tt(0.7, 0.0, 2.0, 0.5, backend=backend, check_pars=True)
        return

    # negative scale parameter
    with pytest.raises(ValueError):
        sr.crps_tt(0.7, 4.7, 2.0, -0.5, backend=backend, check_pars=True)
        return

    # zero scale parameter
    with pytest.raises(ValueError):
        sr.crps_tt(0.7, 4.7, 2.0, 0.0, backend=backend, check_pars=True)
        return

    # lower bound larger than upper bound
    with pytest.raises(ValueError):
        sr.crps_tt(
            0.7, 4.7, 2.0, 0.4, lower=-1, upper=-4, backend=backend, check_pars=True
        )
        return

    ## test correctness

    # default values
    res0 = sr.crps_tt(1.8, 6.9, backend=backend)
    res = sr.crps_tt(1.8, 6.9, 0.0, 1.0, -np.inf, np.inf, backend=backend)
    assert np.isclose(res0, res)

    # single observation
    res = sr.crps_tt(1.8, 6.9, -3.0, 3.3, -5.0, 4.7, backend=backend)
    expected = 2.285149
    assert np.isclose(res, expected)

    # observation below lower bound
    res = sr.crps_tt(-5.8, 6.9, -3.0, 3.3, -5.0, 4.7, backend=backend)
    expected = 2.969029
    assert np.isclose(res, expected)

    # observation above upper bound
    res = sr.crps_tt(7.8, 6.9, -3.0, 3.3, -5.0, 4.7, backend=backend)
    expected = 8.055997
    assert np.isclose(res, expected)

    # aligns with crps_t
    res0 = sr.crps_t(1.8, 6.9, -3.0, 3.3, backend=backend)
    res = sr.crps_tt(1.8, 6.9, -3.0, 3.3, backend=backend)
    assert np.isclose(res, res0)

    # multiple observations
    res = sr.crps_tt(
        np.array([1.8, -2.3]), 6.9, 9.1, 11.1, -4.0, np.inf, backend=backend
    )
    expected = np.array([5.753912, 9.123845])
    assert np.allclose(res, expected)


def test_crps_ct(backend):
    if backend in ["jax", "torch", "tensorflow"]:
        pytest.skip("Not implemented in jax, torch or tensorflow backends")

    ## test shape

    # one observation, multiple forecasts
    res = sr.crps_ct(
        17,
        np.random.uniform(0, 20, (4, 3)),
        np.random.uniform(10, 20, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.crps_ct(
        np.random.uniform(-10, 10, (4, 3)),
        11.1,
        3.2,
        4.1,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.crps_ct(
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
        sr.crps_ct(0.7, -1.7, 2.0, 0.5, backend=backend, check_pars=True)
        return

    # zero df parameter
    with pytest.raises(ValueError):
        sr.crps_ct(0.7, 0.0, 2.0, 0.5, backend=backend, check_pars=True)
        return

    # negative scale parameter
    with pytest.raises(ValueError):
        sr.crps_ct(0.7, 4.7, 2.0, -0.5, backend=backend, check_pars=True)
        return

    # zero scale parameter
    with pytest.raises(ValueError):
        sr.crps_ct(0.7, 4.7, 2.0, 0.0, backend=backend, check_pars=True)
        return

    # lower bound larger than upper bound
    with pytest.raises(ValueError):
        sr.crps_ct(
            0.7, 4.7, 2.0, 0.4, lower=-1, upper=-4, backend=backend, check_pars=True
        )
        return

    ## test correctness

    # default values
    res0 = sr.crps_ct(1.8, 6.9, backend=backend)
    res = sr.crps_ct(1.8, 6.9, 0.0, 1.0, -np.inf, np.inf, backend=backend)
    assert np.isclose(res0, res)

    # single observation
    res = sr.crps_ct(1.8, 6.9, -3.0, 3.3, -5.0, 4.7, backend=backend)
    expected = 2.988432
    assert np.isclose(res, expected)

    # observation below lower bound
    res = sr.crps_ct(-5.8, 6.9, -3.0, 3.3, -5.0, 4.7, backend=backend)
    expected = 1.970734
    assert np.isclose(res, expected)

    # observation above upper bound
    res = sr.crps_ct(7.8, 6.9, -3.0, 3.3, -5.0, 4.7, backend=backend)
    expected = 8.676605
    assert np.isclose(res, expected)

    # aligns with crps_t
    res0 = sr.crps_t(1.8, 6.9, -3.0, 3.3, backend=backend)
    res = sr.crps_ct(1.8, 6.9, -3.0, 3.3, backend=backend)
    assert np.isclose(res, res0)

    # multiple observations
    res = sr.crps_ct(
        np.array([1.8, -2.3]), 6.9, 9.1, 11.1, -4.0, np.inf, backend=backend
    )
    expected = np.array([4.475883, 6.811192])
    assert np.allclose(res, expected)


def test_crps_hypergeometric(backend):
    if backend == "torch":
        pytest.skip("Currently not working in torch backend")

    ## test shape

    # one observation, multiple forecasts
    res = sr.crps_hypergeometric(
        17,
        np.random.randint(10, 20, size=(4, 3)),
        np.random.randint(10, 20, size=(4, 3)),
        np.random.randint(1, 10, size=(4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.crps_hypergeometric(
        np.random.randint(0, 20, size=(4, 3)),
        13,
        4,
        7,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.crps_hypergeometric(
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
        sr.crps_hypergeometric(7, m=-1, n=10, k=5, backend=backend, check_pars=True)
        return

    # negative n parameter
    with pytest.raises(ValueError):
        sr.crps_hypergeometric(7, m=11, n=-2, k=5, backend=backend, check_pars=True)
        return

    # negative k parameter
    with pytest.raises(ValueError):
        sr.crps_hypergeometric(7, m=11, n=10, k=-5, backend=backend, check_pars=True)
        return

    # k > m + n
    with pytest.raises(ValueError):
        sr.crps_hypergeometric(7, m=11, n=10, k=25, backend=backend, check_pars=True)
        return

    # non-integer m parameter
    with pytest.raises(ValueError):
        sr.crps_hypergeometric(7, m=11.1, n=10, k=5, backend=backend, check_pars=True)
        return

    # non-integer n parameter
    with pytest.raises(ValueError):
        sr.crps_hypergeometric(7, m=11, n=10.8, k=5, backend=backend, check_pars=True)
        return

    # non-integer k parameter
    with pytest.raises(ValueError):
        sr.crps_hypergeometric(7, m=11, n=10, k=4.3, backend=backend, check_pars=True)
        return

    ## test correctness

    # single observation
    res = sr.crps_hypergeometric(5, 7, 13, 12, backend=backend)
    expected = 0.4469742
    assert np.isclose(res, expected)

    # negative observation
    res = sr.crps_hypergeometric(-5, 7, 13, 12, backend=backend)
    expected = 8.615395
    assert np.isclose(res, expected)

    # non-integer observation
    res = sr.crps_hypergeometric(5.6, 7, 13, 12, backend=backend)
    expected = 0.9202868
    assert np.isclose(res, expected)

    # multiple observations
    res = sr.crps_hypergeometric(np.array([2, 7.6, -2.1]), 7, 13, 3, backend=backend)
    expected = np.array([0.5965259, 6.1351223, 2.7351223])
    assert np.allclose(res, expected)


def test_crps_laplace(backend):
    ## test shape

    # one observation, multiple forecasts
    res = sr.crps_laplace(
        17,
        np.random.uniform(10, 20, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.crps_laplace(
        np.random.uniform(-10, 10, (4, 3)),
        3.2,
        4.1,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.crps_laplace(
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # negative scale parameter
    with pytest.raises(ValueError):
        sr.crps_laplace(7, 4, -0.1, backend=backend, check_pars=True)
        return

    # zero scale parameter
    with pytest.raises(ValueError):
        sr.crps_laplace(7, 4, 0.0, backend=backend, check_pars=True)
        return

    ## test correctness

    # default location
    res0 = sr.crps_laplace(-3, backend=backend)
    res = sr.crps_laplace(-3, location=0, backend=backend)
    assert np.isclose(res0, res)

    # default scale
    res = sr.crps_laplace(-3, location=0, scale=1.0, backend=backend)
    assert np.isclose(res0, res)

    # invariance to location shifts
    res = sr.crps_laplace(-3 + 0.1, location=0.1, scale=1.0, backend=backend)
    assert np.isclose(res0, res)

    # invariance to rescaling
    res0 = sr.crps_laplace((-3 - 0.1) / 0.8, backend=backend)
    res = sr.crps_laplace(-3, location=0.1, scale=0.8, backend=backend)
    assert np.isclose(res0 * 0.8, res)

    # single observation
    res = sr.crps_laplace(-2.9, 0.1, backend=backend)
    expected = 2.29978707
    assert np.isclose(res, expected)

    # multiple observations
    res = sr.crps_laplace(
        np.array([-2.9, 4.1]),
        np.array([1.1, 1.8]),
        np.array([10.1, 1.2]),
        backend=backend,
    )
    expected = np.array([3.222098, 1.576516])
    assert np.allclose(res, expected)


def test_crps_logis(backend):
    ## test shape

    # one observation, multiple forecasts
    res = sr.crps_logistic(
        17,
        np.random.uniform(10, 20, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.crps_logistic(
        np.random.uniform(-10, 10, (4, 3)),
        3.2,
        4.1,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.crps_logistic(
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # negative scale parameter
    with pytest.raises(ValueError):
        sr.crps_logistic(7, 4, -0.1, backend=backend, check_pars=True)
        return

    # zero scale parameter
    with pytest.raises(ValueError):
        sr.crps_logistic(7, 4, 0.0, backend=backend, check_pars=True)
        return

    ## test correctness

    # default location
    res0 = sr.crps_logistic(-3, backend=backend)
    res = sr.crps_logistic(-3, location=0, backend=backend)
    assert np.isclose(res0, res)

    # default scale
    res = sr.crps_logistic(-3, location=0, scale=1.0, backend=backend)
    assert np.isclose(res0, res)

    # invariance to location shifts
    res = sr.crps_logistic(-3 + 0.1, location=0.1, scale=1.0, backend=backend)
    assert np.isclose(res0, res)

    # invariance to rescaling
    res0 = sr.crps_logistic((-3 - 0.1) / 0.8, backend=backend)
    res = sr.crps_logistic(-3, location=0.1, scale=0.8, backend=backend)
    assert np.isclose(res0 * 0.8, res)

    # single observation
    res = sr.crps_logistic(17.1, 13.8, 3.3, backend=backend)
    expected = 2.067527
    assert np.isclose(res, expected)

    res = sr.crps_logistic(3.1, 4.0, 0.5, backend=backend)
    expected = 0.5529776
    assert np.isclose(res, expected)

    # multiple observations
    res = sr.crps_logistic(
        np.array([-2.9, 4.1]),
        np.array([1.1, 1.8]),
        np.array([10.1, 1.2]),
        backend=backend,
    )
    expected = np.array([4.295051, 1.429361])
    assert np.allclose(res, expected)


def test_crps_loglaplace(backend):
    ## test shape

    # one observation, multiple forecasts
    res = sr.crps_loglaplace(
        17,
        np.random.uniform(10, 20, (4, 3)),
        np.random.uniform(0, 1, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.crps_loglaplace(
        np.random.uniform(-10, 10, (4, 3)),
        3.2,
        0.9,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.crps_loglaplace(
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(0, 1, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # negative scale parameter
    with pytest.raises(ValueError):
        sr.crps_loglaplace(7, 4, -0.1, backend=backend, check_pars=True)
        return

    # zero scale parameter
    with pytest.raises(ValueError):
        sr.crps_loglaplace(7, 4, 0.0, backend=backend, check_pars=True)
        return

    # scale parameter larger than one
    with pytest.raises(ValueError):
        sr.crps_loglaplace(7, 4, 1.1, backend=backend, check_pars=True)
        return

    ## test correctness

    # single observation
    res = sr.crps_loglaplace(7.1, 3.8, 0.3, backend=backend)
    expected = 30.71884
    assert np.isclose(res, expected)

    # negative observation
    res = sr.crps_loglaplace(-4.1, -3.8, 0.3, backend=backend)
    expected = 4.118925
    assert np.isclose(res, expected)

    # multiple observations
    res = sr.crps_loglaplace(
        np.array([2.9, 4.4]),
        np.array([1.1, 1.8]),
        np.array([0.1, 0.9]),
        backend=backend,
    )
    expected = np.array([0.09159719, 1.95400976])
    assert np.allclose(res, expected)


def test_crps_loglogistic(backend):
    if backend == "torch":
        pytest.skip("Not implemented in torch backend")

    ## test shape

    # one observation, multiple forecasts
    res = sr.crps_loglogistic(
        17,
        np.random.uniform(10, 20, (4, 3)),
        np.random.uniform(0, 1, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.crps_loglogistic(
        np.random.uniform(-10, 10, (4, 3)),
        3.2,
        0.9,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.crps_loglogistic(
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(0, 1, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # negative scale parameter
    with pytest.raises(ValueError):
        sr.crps_loglogistic(7, 4, -0.1, backend=backend, check_pars=True)
        return

    # zero scale parameter
    with pytest.raises(ValueError):
        sr.crps_loglogistic(7, 4, 0.0, backend=backend, check_pars=True)
        return

    # scale parameter larger than one
    with pytest.raises(ValueError):
        sr.crps_loglogistic(7, 4, 1.1, backend=backend, check_pars=True)
        return

    ## test correctness

    # single observation
    res = sr.crps_loglogistic(7.1, 3.8, 0.3, backend=backend)
    expected = 29.35987
    assert np.isclose(res, expected)

    # negative observation
    res = sr.crps_loglogistic(-4.1, -3.8, 0.3, backend=backend)
    expected = 4.118243
    assert np.isclose(res, expected)

    # multiple observations
    res = sr.crps_loglogistic(
        np.array([2.9, 4.4]),
        np.array([1.1, 1.8]),
        np.array([0.1, 0.9]),
        backend=backend,
    )
    expected = np.array([0.1256149, 3.1692104])
    assert np.allclose(res, expected)


def test_crps_lognormal(backend):
    ## test shape

    # one observation, multiple forecasts
    res = sr.crps_lognormal(
        17,
        np.random.uniform(10, 20, (4, 3)),
        np.random.uniform(0, 10, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.crps_lognormal(
        np.random.uniform(-10, 10, (4, 3)),
        3.2,
        2.9,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.crps_lognormal(
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(0, 1, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # negative scale parameter
    with pytest.raises(ValueError):
        sr.crps_lognormal(7, 4, -0.1, backend=backend, check_pars=True)
        return

    # zero scale parameter
    with pytest.raises(ValueError):
        sr.crps_lognormal(7, 4, 0.0, backend=backend, check_pars=True)
        return

    ## test correctness

    # single observation
    res = sr.crps_lognormal(7.1, 3.8, 0.3, backend=backend)
    expected = 31.80341
    assert np.isclose(res, expected)

    # negative observation
    res = sr.crps_lognormal(-4.1, -3.8, 0.3, backend=backend)
    expected = 4.119469
    assert np.isclose(res, expected)

    # multiple observations
    res = sr.crps_lognormal(
        np.array([2.9, 4.4]),
        np.array([1.1, 1.8]),
        np.array([0.1, 0.9]),
        backend=backend,
    )
    expected = np.array([0.08472861, 1.63688414])
    assert np.allclose(res, expected)


def test_crps_mixnorm(backend):
    ## test shape

    # one observation, multiple forecasts
    res = sr.crps_mixnorm(
        17,
        np.random.uniform(-5, 5, (4, 3, 5)),
        np.random.uniform(0, 5, (4, 3, 5)),
        np.random.uniform(0, 1, (4, 3, 5)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.crps_mixnorm(
        np.random.uniform(-5, 5, (4, 3)),
        np.array([0.0, -2.9, 0.9, 3.2, 7.0]),
        np.array([0.6, 2.9, 0.9, 3.2, 7.0]),
        np.array([0.6, 2.9, 0.9, 0.2, 0.1]),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.crps_mixnorm(
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
        sr.crps_mixnorm(
            7, np.array([4, 2]), np.array([-1, 2]), backend=backend, check_pars=True
        )
        return

    # negative weight parameter
    with pytest.raises(ValueError):
        sr.crps_mixnorm(
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
    res = sr.crps_mixnorm(obs, m, s, w, backend=backend)
    expected = 0.4510451
    assert np.isclose(res, expected)

    # default weights
    res0 = sr.crps_mixnorm(obs, m, s, backend=backend)
    assert np.isclose(res, res0)

    # non-constant weights
    w = [0.3, 0.1, 0.6]
    res = sr.crps_mixnorm(obs, m, s, w, backend=backend)
    expected = 0.2354619
    assert np.isclose(res, expected)

    # weights that do not sum to one
    w = [8.3, 0.1, 4.6]
    res = sr.crps_mixnorm(obs, m, s, w, backend=backend)
    expected = 0.1678256
    assert np.isclose(res, expected)

    # m-axis argument
    obs = [-1.6, 0.3]
    m = [[0.0, -2.9], [0.6, 0.0], [-1.1, -2.3]]
    s = [[0.5, 1.7], [1.1, 0.7], [1.4, 1.5]]
    res1 = sr.crps_mixnorm(obs, m, s, m_axis=0, backend=backend)

    m = [[0.0, 0.6, -1.1], [-2.9, 0.0, -2.3]]
    s = [[0.5, 1.1, 1.4], [1.7, 0.7, 1.5]]
    res2 = sr.crps_mixnorm(obs, m, s, backend=backend)
    assert np.allclose(res1, res2)

    # check equality with normal distribution
    obs = 1.6
    m, s, w = [1.8, 1.8], [2.3, 2.3], [0.6, 0.8]
    res0 = sr.crps_normal(obs, m[0], s[0], backend=backend)
    res = sr.crps_mixnorm(obs, m, s, w, backend=backend)
    assert np.isclose(res0, res)


def test_crps_negbinom(backend):
    if backend in ["jax", "torch", "tensorflow"]:
        pytest.skip("Not implemented in jax, torch or tensorflow backends")

    ## test shape

    # one observation, multiple forecasts
    res = sr.crps_negbinom(
        17,
        np.random.uniform(10, 20, (4, 3)),
        np.random.uniform(0, 1, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.crps_negbinom(
        np.random.uniform(0, 20, (4, 3)),
        12,
        0.2,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.crps_negbinom(
        np.random.uniform(0, 20, (4, 3)),
        np.random.uniform(10, 20, (4, 3)),
        np.random.uniform(0, 1, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # prob and mu are both provided
    with pytest.raises(ValueError):
        sr.crps_negbinom(0.3, 7.0, 0.8, mu=7.3, backend=backend)

    # neither prob nor mu are provided
    with pytest.raises(ValueError):
        sr.crps_negbinom(0.3, 8, backend=backend)

    # negative size parameter
    with pytest.raises(ValueError):
        sr.crps_negbinom(4.3, -8, 0.8, backend=backend, check_pars=True)

    # negative mu parameter
    with pytest.raises(ValueError):
        sr.crps_negbinom(4.3, 8, mu=-0.8, backend=backend, check_pars=True)

    # negative prob parameter
    with pytest.raises(ValueError):
        sr.crps_negbinom(4.3, 8, -0.8, backend=backend, check_pars=True)

    # prob parameter larger than one
    with pytest.raises(ValueError):
        sr.crps_negbinom(4.3, 8, 1.8, backend=backend, check_pars=True)

    ## test correctness

    # single observation
    res = sr.crps_negbinom(2.0, 7.0, 0.8, backend=backend)
    expected = 0.3834322
    assert np.isclose(res, expected)

    # non-integer size
    res = sr.crps_negbinom(1.5, 2.4, 0.5, backend=backend)
    expected = 0.5423484
    assert np.isclose(res, expected)

    # negative observation
    res = sr.crps_negbinom(-1.0, 17.0, 0.1, backend=backend)
    expected = 132.0942
    assert np.isclose(res, expected)

    # mu given instead of prob
    res = sr.crps_negbinom(2.3, 11.0, mu=7.3, backend=backend)
    expected = 3.149218
    assert np.isclose(res, expected)

    # multiple observations
    res = sr.crps_negbinom(np.array([1.9, 7.3]), 7.0, 0.8, backend=backend)
    expected = np.array([0.3827689, 4.7630042])
    assert np.allclose(res, expected)


def test_crps_normal(backend):
    ## test shape

    # one observation, multiple forecasts
    res = sr.crps_normal(
        17,
        np.random.uniform(10, 20, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.crps_normal(
        np.random.uniform(-10, 10, (4, 3)),
        3.2,
        4.1,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.crps_normal(
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # negative scale parameter
    with pytest.raises(ValueError):
        sr.crps_normal(7, 4, -0.1, backend=backend, check_pars=True)
        return

    # zero scale parameter
    with pytest.raises(ValueError):
        sr.crps_normal(7, 4, 0.0, backend=backend, check_pars=True)
        return

    ## test correctness

    # default mu
    res0 = sr.crps_normal(-3, backend=backend)
    res = sr.crps_normal(-3, mu=0, backend=backend)
    assert np.isclose(res0, res)

    # default sigma
    res = sr.crps_normal(-3, mu=0, sigma=1.0, backend=backend)
    assert np.isclose(res0, res)

    # invariance to location shifts
    res = sr.crps_normal(-3 + 0.1, mu=0.1, sigma=1.0, backend=backend)
    assert np.isclose(res0, res)

    # invariance to rescaling
    res0 = sr.crps_normal((-3 - 0.1) / 0.8, backend=backend)
    res = sr.crps_normal(-3, mu=0.1, sigma=0.8, backend=backend)
    assert np.isclose(res0 * 0.8, res)

    # single observation
    res = sr.crps_normal(7.1, 3.8, 0.3, backend=backend)
    expected = 3.130743
    assert np.isclose(res, expected)

    res = sr.crps_normal(3.1, 4.0, 0.5, backend=backend)
    expected = 0.6321808
    assert np.isclose(res, expected)

    # multiple observations
    res = sr.crps_normal(
        np.array([-2.9, 4.4]),
        np.array([1.1, 1.8]),
        np.array([10.1, 1.2]),
        backend=backend,
    )
    expected = np.array([2.984174, 1.935862])
    assert np.allclose(res, expected)


def test_crps_2pnormal(backend):
    ## test shape

    # one observation, multiple forecasts
    res = sr.crps_2pnormal(
        17,
        np.random.uniform(10, 20, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.crps_2pnormal(
        np.random.uniform(-10, 10, (4, 3)),
        3.2,
        4.1,
        2.8,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.crps_2pnormal(
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(-10, 10, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # negative scale1 parameter
    with pytest.raises(ValueError):
        sr.crps_2pnormal(7, -0.1, 0.1, backend=backend, check_pars=True)
        return

    # zero scale1 parameter
    with pytest.raises(ValueError):
        sr.crps_2pnormal(7, 0.0, backend=backend, check_pars=True)
        return

    # negative scale2 parameter
    with pytest.raises(ValueError):
        sr.crps_2pnormal(7, 0.1, -2.1, backend=backend, check_pars=True)
        return

    # zero scale2 parameter
    with pytest.raises(ValueError):
        sr.crps_2pnormal(7, 1.1, 0.0, backend=backend, check_pars=True)
        return

    ## test correctness

    # default scale1
    res0 = sr.crps_2pnormal(-3, backend=backend)
    res = sr.crps_2pnormal(-3, scale1=1.0, backend=backend)
    assert np.isclose(res0, res)

    # default scale2
    res = sr.crps_2pnormal(-3, scale1=1.0, scale2=1.0, backend=backend)
    assert np.isclose(res0, res)

    # default location
    res = sr.crps_2pnormal(-3, scale1=1.0, scale2=1.0, location=0.0, backend=backend)
    assert np.isclose(res0, res)

    # invariance to location shifts
    res = sr.crps_2pnormal(-3 + 0.1, 1.0, 1.0, location=0.1, backend=backend)
    assert np.isclose(res0, res)

    # single observation
    res = sr.crps_2pnormal(7.1, 3.8, 0.3, -2.0, backend=backend)
    expected = 10.5914
    assert np.isclose(res, expected)

    # check equivalence with standard normal distribution
    res0 = sr.crps_normal(3.1, 4.0, 0.5, backend=backend)
    res = sr.crps_2pnormal(3.1, 0.5, 0.5, 4.0, backend=backend)
    assert np.isclose(res0, res)

    # multiple observations
    res = sr.crps_2pnormal(
        np.array([-2.9, 4.4]),
        np.array([1.1, 1.8]),
        np.array([10.1, 1.2]),
        backend=backend,
    )
    expected = np.array([6.572028, 4.026696])
    assert np.allclose(res, expected)


def test_crps_poisson(backend):
    ## test shape

    # one observation, multiple forecasts
    res = sr.crps_poisson(
        17,
        np.random.uniform(0, 20, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.crps_poisson(
        np.random.uniform(-5, 10, (4, 3)),
        3.2,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.crps_poisson(
        np.random.uniform(-5, 10, (4, 3)),
        np.random.uniform(0, 20, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # negative mean parameter
    with pytest.raises(ValueError):
        sr.crps_poisson(7, -0.1, backend=backend, check_pars=True)
        return

    # zero mean parameter
    with pytest.raises(ValueError):
        sr.crps_poisson(7, 0.0, backend=backend, check_pars=True)
        return

    ## test correctness

    # single observation
    res = sr.crps_poisson(1.0, 3.0, backend=backend)
    expected = 1.143447
    assert np.isclose(res, expected)

    res = sr.crps_poisson(1.5, 2.3, backend=backend)
    expected = 0.5001159
    assert np.isclose(res, expected)

    # negative observation
    res = sr.crps_poisson(-1.0, 1.5, backend=backend)
    expected = 1.840259
    assert np.isclose(res, expected)

    # multiple observations
    res = sr.crps_poisson(np.array([-2.9, 4.4]), np.array([10.1, 1.2]), backend=backend)
    expected = np.array([11.218179, 2.630758])
    assert np.allclose(res, expected)


def test_crps_t(backend):
    if backend == "torch":
        pytest.skip("Not implemented in torch backend")

    ## test shape

    # one observation, multiple forecasts
    res = sr.crps_t(
        17,
        np.random.uniform(0, 20, (4, 3)),
        np.random.uniform(10, 20, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.crps_t(
        np.random.uniform(-5, 20, (4, 3)),
        12,
        12,
        12,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.crps_t(
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
        sr.crps_t(7, -4, backend=backend, check_pars=True)
        return

    # zero df parameter
    with pytest.raises(ValueError):
        sr.crps_t(7, 0, backend=backend, check_pars=True)
        return

    # negative scale parameter
    with pytest.raises(ValueError):
        sr.crps_t(7, 4, scale=-0.1, backend=backend, check_pars=True)
        return

    # zero scale parameter
    with pytest.raises(ValueError):
        sr.crps_t(7, 4, scale=0.0, backend=backend, check_pars=True)
        return

    ## test correctness

    # default location
    res0 = sr.crps_t(-3, 2.1, backend=backend)
    res = sr.crps_t(-3, 2.1, location=0, backend=backend)
    assert np.isclose(res0, res)

    # default scale
    res = sr.crps_t(-3, 2.1, location=0, scale=1.0, backend=backend)
    assert np.isclose(res0, res)

    # invariance to location shifts
    res = sr.crps_t(-3 + 0.1, 2.1, location=0.1, scale=1.0, backend=backend)
    assert np.isclose(res0, res)

    # invariance to rescaling
    res0 = sr.crps_t((-3 - 0.1) / 0.8, 2.1, backend=backend)
    res = sr.crps_t(-3, 2.1, location=0.1, scale=0.8, backend=backend)
    assert np.isclose(res0 * 0.8, res)

    # single observation
    res = sr.crps_t(7.1, 4.2, 3.8, 0.3, backend=backend)
    expected = 3.082766
    assert np.isclose(res, expected)

    # negative observation
    res = sr.crps_t(-3.1, 2.9, 4.0, 0.5, backend=backend)
    expected = 6.682638
    assert np.isclose(res, expected)

    # multiple observations
    res = sr.crps_t(
        np.array([-2.9, 4.4]),
        np.array([5.1, 19.4]),
        np.array([1.1, 1.8]),
        np.array([10.1, 1.2]),
        backend=backend,
    )
    expected = np.array([3.183587, 1.914685])
    assert np.allclose(res, expected)


def test_crps_uniform(backend):
    ## test shape

    # one observation, multiple forecasts
    res = sr.crps_uniform(
        7,
        np.random.uniform(0, 5, (4, 3)),
        np.random.uniform(5, 10, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, one forecast
    res = sr.crps_uniform(
        np.random.uniform(0, 10, (4, 3)),
        1,
        7,
        backend=backend,
    )
    assert res.shape == (4, 3)

    # multiple observations, multiple forecasts
    res = sr.crps_uniform(
        np.random.uniform(0, 10, (4, 3)),
        np.random.uniform(0, 5, (4, 3)),
        np.random.uniform(5, 10, (4, 3)),
        backend=backend,
    )
    assert res.shape == (4, 3)

    ## test exceptions

    # negative lmass parameter
    with pytest.raises(ValueError):
        sr.crps_uniform(0.7, lmass=-0.1, backend=backend, check_pars=True)
        return

    # lmass parameter greater than one
    with pytest.raises(ValueError):
        sr.crps_uniform(0.7, lmass=4, backend=backend, check_pars=True)
        return

    # negative umass parameter
    with pytest.raises(ValueError):
        sr.crps_uniform(0.7, umass=-0.1, backend=backend, check_pars=True)
        return

    # umass parameter greater than one
    with pytest.raises(ValueError):
        sr.crps_uniform(0.7, umass=4, backend=backend, check_pars=True)
        return

    # lmass + umass >= 1
    with pytest.raises(ValueError):
        sr.crps_uniform(0.7, lmass=0.2, umass=0.9, backend=backend, check_pars=True)
        return

    # min >= max
    with pytest.raises(ValueError):
        sr.crps_uniform(0.7, min=2, max=1, backend=backend, check_pars=True)
        return

    ## test correctness

    # default min
    res0 = sr.crps_uniform(0.7, backend=backend)
    res = sr.crps_uniform(0.7, min=0, backend=backend)
    assert np.isclose(res0, res)

    # default sigma
    res = sr.crps_uniform(0.7, min=0, max=1.0, backend=backend)
    assert np.isclose(res0, res)

    # invariance to location shifts
    res = sr.crps_uniform(0.7 + 0.1, min=0.1, max=1.1, backend=backend)
    assert np.isclose(res0, res)

    # single observation
    res = sr.crps_uniform(0.3, -1.0, 2.1, 0.3, 0.1, backend=backend)
    expected = 0.3960968
    assert np.isclose(res, expected)

    res = sr.crps_uniform(2.2, 0.1, 3.1, backend=backend)
    expected = 0.37
    assert np.isclose(res, expected)

    # observation outside of interval range
    res = sr.crps_uniform(-17.9, -15.2, -8.7, 0.2, backend=backend)
    expected = 4.086667
    assert np.isclose(res, expected)

    # multiple observations
    res = sr.crps_uniform(
        np.array([-2.9, 4.4]),
        np.array([1.1, 1.8]),
        np.array([10.1, 2.1]),
        backend=backend,
    )
    expected = np.array([7.0, 2.4])
    assert np.allclose(res, expected)
