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

    # test shapes
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

    # approx zero when perfect forecast
    perfect_fct = obs[..., None] + np.random.randn(N, ENSEMBLE_SIZE) * 0.00001
    res = sr.crps_ensemble(obs, perfect_fct, estimator=estimator, backend=backend)
    res = np.asarray(res)
    assert not np.any(res - 0.0 > 0.0001)


def test_crps_estimators(backend):
    obs = np.random.randn(N)
    mu = obs + np.random.randn(N) * 0.3
    sigma = abs(np.random.randn(N)) * 0.5
    fct = np.random.randn(N, ENSEMBLE_SIZE) * sigma[..., None] + mu[..., None]

    # equality of estimators
    res_nrg = sr.crps_ensemble(obs, fct, estimator="nrg", backend=backend)
    res_fair = sr.crps_ensemble(obs, fct, estimator="fair", backend=backend)
    res_pwm = sr.crps_ensemble(obs, fct, estimator="pwm", backend=backend)
    res_qd = sr.crps_ensemble(obs, fct, estimator="qd", backend=backend)
    res_int = sr.crps_ensemble(obs, fct, estimator="int", backend=backend)

    assert np.allclose(res_nrg, res_qd)
    assert np.allclose(res_nrg, res_int)
    assert np.allclose(res_fair, res_pwm)


def test_crps_quantile(backend):
    # test shapes
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

    res = sr.crps_beta(
        np.random.uniform(0, 1, (3, 3)),
        np.random.uniform(0, 3, (3, 3)),
        1.1,
        backend=backend,
    )
    assert res.shape == (3, 3)
    assert not np.any(np.isnan(res))

    # test exceptions
    with pytest.raises(ValueError):
        sr.crps_beta(0.3, 0.7, 1.1, lower=1.0, upper=0.0, backend=backend)
        return

    # correctness tests
    res = sr.crps_beta(0.3, 0.7, 1.1, backend=backend)
    expected = 0.0850102437
    assert np.isclose(res, expected)

    res = sr.crps_beta(-3.0, 0.7, 1.1, lower=-5.0, upper=4.0, backend=backend)
    expected = 0.883206751
    assert np.isclose(res, expected)

    # test when lower and upper are arrays
    res = sr.crps_beta(
        -3.0,
        0.7,
        1.1,
        lower=np.array([-5.0, -5.0]),
        upper=np.array([4.0, 4.0]),
    )
    expected = np.array([0.883206751, 0.883206751])
    assert np.allclose(res, expected)


def test_crps_binomial(backend):
    if backend == "torch":
        pytest.skip("Not implemented in torch backend")

    # test correctness
    res = sr.crps_binomial(8, 10, 0.9, backend=backend)
    expected = 0.6685115
    assert np.isclose(res, expected)

    res = sr.crps_binomial(-8, 10, 0.9, backend=backend)
    expected = 16.49896
    assert np.isclose(res, expected)

    res = sr.crps_binomial(18, 10, 0.9, backend=backend)
    expected = 8.498957
    assert np.isclose(res, expected)

    # test broadcasting
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


def test_crps_exponential(backend):
    # TODO: add and test exception handling

    # test correctness
    obs, rate = 3, 0.7
    res = sr.crps_exponential(obs, rate, backend=backend)
    expected = 1.20701837
    assert np.isclose(res, expected)


def test_crps_exponentialM(backend):
    obs, mass, location, scale = 0.3, 0.1, 0.0, 1.0
    res = sr.crps_exponentialM(obs, mass, location, scale, backend=backend)
    expected = 0.2384728
    assert np.isclose(res, expected)

    obs, mass, location, scale = 0.3, 0.1, -2.0, 3.0
    res = sr.crps_exponentialM(obs, mass, location, scale, backend=backend)
    expected = 0.6236187
    assert np.isclose(res, expected)

    obs, mass, location, scale = -1.2, 0.1, -2.0, 3.0
    res = sr.crps_exponentialM(obs, mass, location, scale, backend=backend)
    expected = 0.751013
    assert np.isclose(res, expected)


def test_crps_2pexponential(backend):
    obs, scale1, scale2, location = 0.3, 0.1, 4.3, 0.0
    res = sr.crps_2pexponential(obs, scale1, scale2, location, backend=backend)
    expected = 1.787032
    assert np.isclose(res, expected)

    obs, scale1, scale2, location = -20.8, 7.1, 2.0, -25.4
    res = sr.crps_2pexponential(obs, scale1, scale2, location, backend=backend)
    expected = 6.018359
    assert np.isclose(res, expected)


def test_crps_gamma(backend):
    obs, shape, rate = 0.2, 1.1, 0.7
    expected = 0.6343718

    res = sr.crps_gamma(obs, shape, rate, backend=backend)
    assert np.isclose(res, expected)

    res = sr.crps_gamma(obs, shape, scale=1 / rate, backend=backend)
    assert np.isclose(res, expected)

    with pytest.raises(ValueError):
        sr.crps_gamma(obs, shape, rate, scale=1 / rate, backend=backend)
        return

    with pytest.raises(ValueError):
        sr.crps_gamma(obs, shape, backend=backend)
        return


def test_crps_csg0(backend):
    obs, shape, rate, shift = 0.7, 0.5, 2.0, 0.3
    expected = 0.5411044

    expected_gamma = sr.crps_gamma(obs, shape, rate, backend=backend)
    res_gamma = sr.crps_csg0(obs, shape=shape, rate=rate, shift=0.0, backend=backend)
    assert np.isclose(res_gamma, expected_gamma)

    res = sr.crps_csg0(obs, shape=shape, rate=rate, shift=shift, backend=backend)
    assert np.isclose(res, expected)

    res = sr.crps_csg0(obs, shape=shape, scale=1.0 / rate, shift=shift, backend=backend)
    assert np.isclose(res, expected)

    with pytest.raises(ValueError):
        sr.crps_csg0(
            obs, shape=shape, rate=rate, scale=1.0 / rate, shift=shift, backend=backend
        )
        return

    with pytest.raises(ValueError):
        sr.crps_csg0(obs, shape=shape, shift=shift, backend=backend)
        return


def test_crps_gev(backend):
    if backend == "torch":
        pytest.skip("`expi` not implemented in torch backend")

    obs, xi, mu, sigma = 0.3, 0.0, 0.0, 1.0
    assert np.isclose(sr.crps_gev(obs, xi, backend=backend), 0.276440963)
    mu = 0.1
    assert np.isclose(
        sr.crps_gev(obs + mu, xi, location=mu, backend=backend), 0.276440963
    )
    sigma = 0.9
    mu = 0.0
    assert np.isclose(
        sr.crps_gev(obs * sigma, xi, scale=sigma, backend=backend),
        0.276440963 * sigma,
    )

    obs, xi, mu, sigma = 0.3, 0.7, 0.0, 1.0
    assert np.isclose(sr.crps_gev(obs, xi, backend=backend), 0.458044365)
    mu = 0.1
    assert np.isclose(
        sr.crps_gev(obs + mu, xi, location=mu, backend=backend), 0.458044365
    )
    sigma = 0.9
    mu = 0.0
    assert np.isclose(
        sr.crps_gev(obs * sigma, xi, scale=sigma, backend=backend),
        0.458044365 * sigma,
    )

    obs, xi, mu, sigma = 0.3, -0.7, 0.0, 1.0
    assert np.isclose(sr.crps_gev(obs, xi, backend=backend), 0.207621488)
    mu = 0.1
    assert np.isclose(
        sr.crps_gev(obs + mu, xi, location=mu, backend=backend), 0.207621488
    )
    sigma = 0.9
    mu = 0.0
    assert np.isclose(
        sr.crps_gev(obs * sigma, xi, scale=sigma, backend=backend),
        0.207621488 * sigma,
    )


def test_crps_gpd(backend):
    assert np.isclose(sr.crps_gpd(0.3, 0.9, backend=backend), 0.6849332)
    assert np.isclose(sr.crps_gpd(-0.3, 0.9, backend=backend), 1.209091)
    assert np.isclose(sr.crps_gpd(0.3, -0.9, backend=backend), 0.1338672)
    assert np.isclose(sr.crps_gpd(-0.3, -0.9, backend=backend), 0.6448276)

    assert np.isnan(sr.crps_gpd(0.3, 1.0, backend=backend))
    assert np.isnan(sr.crps_gpd(0.3, 1.2, backend=backend))
    assert np.isnan(sr.crps_gpd(0.3, 0.9, mass=-0.1, backend=backend))
    assert np.isnan(sr.crps_gpd(0.3, 0.9, mass=1.1, backend=backend))

    res = 0.281636441
    assert np.isclose(sr.crps_gpd(0.3 + 0.1, 0.0, location=0.1, backend=backend), res)
    assert np.isclose(
        sr.crps_gpd(0.3 * 0.9, 0.0, scale=0.9, backend=backend), res * 0.9
    )


def test_crps_gtclogis(backend):
    obs, location, scale, lower, upper, lmass, umass = (
        1.8,
        -3.0,
        3.3,
        -5.0,
        4.7,
        0.1,
        0.15,
    )
    expected = 1.599721
    res = sr.crps_gtclogistic(
        obs, location, scale, lower, upper, lmass, umass, backend=backend
    )
    assert np.isclose(res, expected)

    # aligns with crps_logistic
    res0 = sr.crps_logistic(obs, location, scale, backend=backend)
    res = sr.crps_gtclogistic(obs, location, scale, backend=backend)
    assert np.isclose(res, res0)

    # aligns with crps_tlogistic
    res0 = sr.crps_tlogistic(obs, location, scale, lower, upper, backend=backend)
    res = sr.crps_gtclogistic(obs, location, scale, lower, upper, backend=backend)
    assert np.isclose(res, res0)


def test_crps_tlogis(backend):
    obs, location, scale, lower, upper = 4.9, 3.5, 2.3, 0.0, 20.0
    expected = 0.7658979
    res = sr.crps_tlogistic(obs, location, scale, lower, upper, backend=backend)
    assert np.isclose(res, expected)

    # aligns with crps_logistic
    res0 = sr.crps_logistic(obs, location, scale, backend=backend)
    res = sr.crps_tlogistic(obs, location, scale, backend=backend)
    assert np.isclose(res, res0)


def test_crps_clogis(backend):
    obs, location, scale, lower, upper = -0.9, 0.4, 1.1, 0.0, 1.0
    expected = 1.13237
    res = sr.crps_clogistic(obs, location, scale, lower, upper, backend=backend)
    assert np.isclose(res, expected)

    # aligns with crps_logistic
    res0 = sr.crps_logistic(obs, location, scale, backend=backend)
    res = sr.crps_clogistic(obs, location, scale, backend=backend)
    assert np.isclose(res, res0)


def test_crps_gtcnormal(backend):
    obs, location, scale, lower, upper, lmass, umass = (
        0.9,
        -2.3,
        4.1,
        -7.3,
        1.7,
        0.0,
        0.21,
    )
    expected = 1.422805
    res = sr.crps_gtcnormal(
        obs, location, scale, lower, upper, lmass, umass, backend=backend
    )
    assert np.isclose(res, expected)

    # aligns with crps_normal
    res0 = sr.crps_normal(obs, location, scale, backend=backend)
    res = sr.crps_gtcnormal(obs, location, scale, backend=backend)
    assert np.isclose(res, res0)

    # aligns with crps_tnormal
    res0 = sr.crps_tnormal(obs, location, scale, lower, upper, backend=backend)
    res = sr.crps_gtcnormal(obs, location, scale, lower, upper, backend=backend)
    assert np.isclose(res, res0)


def test_crps_tnormal(backend):
    obs, location, scale, lower, upper = -1.0, 2.9, 2.2, 1.5, 17.3
    expected = 3.982434
    res = sr.crps_tnormal(obs, location, scale, lower, upper, backend=backend)
    assert np.isclose(res, expected)

    # aligns with crps_normal
    res0 = sr.crps_normal(obs, location, scale, backend=backend)
    res = sr.crps_tnormal(obs, location, scale, backend=backend)
    assert np.isclose(res, res0)


def test_crps_cnormal(backend):
    obs, location, scale, lower, upper = 1.8, 0.4, 1.1, 0.0, 2.0
    expected = 0.8296078
    res = sr.crps_cnormal(obs, location, scale, lower, upper, backend=backend)
    assert np.isclose(res, expected)

    # aligns with crps_normal
    res0 = sr.crps_normal(obs, location, scale, backend=backend)
    res = sr.crps_cnormal(obs, location, scale, backend=backend)
    assert np.isclose(res, res0)


def test_crps_gtct(backend):
    if backend in ["jax", "torch", "tensorflow"]:
        pytest.skip("Not implemented in jax, torch or tensorflow backends")
    obs, df, location, scale, lower, upper, lmass, umass = (
        0.9,
        20.1,
        -2.3,
        4.1,
        -7.3,
        1.7,
        0.0,
        0.21,
    )
    expected = 1.423042
    res = sr.crps_gtct(
        obs, df, location, scale, lower, upper, lmass, umass, backend=backend
    )
    assert np.isclose(res, expected)

    # aligns with crps_t
    res0 = sr.crps_t(obs, df, location, scale, backend=backend)
    res = sr.crps_gtct(obs, df, location, scale, backend=backend)
    assert np.isclose(res, res0)

    # aligns with crps_tnormal
    res0 = sr.crps_tt(obs, df, location, scale, lower, upper, backend=backend)
    res = sr.crps_gtct(obs, df, location, scale, lower, upper, backend=backend)
    assert np.isclose(res, res0)


def test_crps_tt(backend):
    if backend in ["jax", "torch", "tensorflow"]:
        pytest.skip("Not implemented in jax, torch or tensorflow backends")

    obs, df, location, scale, lower, upper = -1.0, 2.9, 3.1, 4.2, 1.5, 17.3
    expected = 5.084272
    res = sr.crps_tt(obs, df, location, scale, lower, upper, backend=backend)
    assert np.isclose(res, expected)

    # aligns with crps_t
    res0 = sr.crps_t(obs, df, location, scale, backend=backend)
    res = sr.crps_tt(obs, df, location, scale, backend=backend)
    assert np.isclose(res, res0)


def test_crps_ct(backend):
    if backend in ["jax", "torch", "tensorflow"]:
        pytest.skip("Not implemented in jax, torch or tensorflow backends")

    obs, df, location, scale, lower, upper = 1.8, 5.4, 0.4, 1.1, 0.0, 2.0
    expected = 0.8028996
    res = sr.crps_ct(obs, df, location, scale, lower, upper, backend=backend)
    assert np.isclose(res, expected)

    # aligns with crps_t
    res0 = sr.crps_t(obs, df, location, scale, backend=backend)
    res = sr.crps_ct(obs, df, location, scale, backend=backend)
    assert np.isclose(res, res0)


def test_crps_hypergeometric(backend):
    if backend == "torch":
        pytest.skip("Currently not working in torch backend")

    # test shapes
    res = sr.crps_hypergeometric(5 * np.ones((2, 2)), 7, 13, 12, backend=backend)
    assert res.shape == (2, 2)

    res = sr.crps_hypergeometric(5, 7 * np.ones((2, 2)), 13, 12, backend=backend)
    assert res.shape == (2, 2)

    res = sr.crps_hypergeometric(5, 7, 13 * np.ones((2, 2)), 12, backend=backend)
    assert res.shape == (2, 2)

    # test correctness
    assert np.isclose(sr.crps_hypergeometric(5, 7, 13, 12), 0.4469742)


def test_crps_laplace(backend):
    assert np.isclose(sr.crps_laplace(-3, backend=backend), 2.29978707)
    assert np.isclose(
        sr.crps_laplace(-3 + 0.1, location=0.1, backend=backend), 2.29978707
    )
    assert np.isclose(
        sr.crps_laplace(-3 * 0.9, scale=0.9, backend=backend), 0.9 * 2.29978707
    )


def test_crps_logis(backend):
    obs, mu, sigma = 17.1, 13.8, 3.3
    expected = 2.067527
    res = sr.crps_logistic(obs, mu, sigma, backend=backend)
    assert np.isclose(res, expected)

    obs, mu, sigma = 3.1, 4.0, 0.5
    expected = 0.5529776
    res = sr.crps_logistic(obs, mu, sigma, backend=backend)
    assert np.isclose(res, expected)


def test_crps_loglaplace(backend):
    assert np.isclose(sr.crps_loglaplace(3.0, 0.1, 0.9, backend=backend), 1.16202051)


def test_crps_loglogistic(backend):
    if backend == "torch":
        pytest.skip("Not implemented in torch backend")

    # TODO: investigate why JAX results are different from other backends
    # (would fail test with smaller tolerance)
    assert np.isclose(
        sr.crps_loglogistic(3.0, 0.1, 0.9, backend=backend), 1.13295277, atol=1e-4
    )


def test_crps_lognormal(backend):
    obs = np.exp(np.random.randn(N))
    mulog = np.log(obs) + np.random.randn(N) * 0.1
    sigmalog = abs(np.random.randn(N)) * 0.3

    # non-negative values
    res = sr.crps_lognormal(obs, mulog, sigmalog, backend=backend)
    res = np.asarray(res)
    assert not np.any(np.isnan(res))
    assert not np.any(res < 0.0)

    # approx zero when perfect forecast
    mulog = np.log(obs) + np.random.randn(N) * 1e-6
    sigmalog = abs(np.random.randn(N)) * 1e-6
    res = sr.crps_lognormal(obs, mulog, sigmalog, backend=backend)
    res = np.asarray(res)
    assert not np.any(np.isnan(res))
    assert not np.any(res - 0.0 > 0.0001)


def test_crps_mixnorm(backend):
    obs, m, s, w = 0.3, [0.0, -2.9, 0.9], [0.5, 1.4, 0.7], [1 / 3, 1 / 3, 1 / 3]
    res = sr.crps_mixnorm(obs, m, s, w, backend=backend)
    expected = 0.4510451
    assert np.isclose(res, expected)

    res0 = sr.crps_mixnorm(obs, m, s, backend=backend)
    assert np.isclose(res, res0)

    w = [0.3, 0.1, 0.6]
    res = sr.crps_mixnorm(obs, m, s, w, backend=backend)
    expected = 0.2354619
    assert np.isclose(res, expected)

    obs = [-1.6, 0.3]
    m = [[0.0, -2.9], [0.6, 0.0], [-1.1, -2.3]]
    s = [[0.5, 1.7], [1.1, 0.7], [1.4, 1.5]]
    res1 = sr.crps_mixnorm(obs, m, s, m_axis=0, backend=backend)

    m = [[0.0, 0.6, -1.1], [-2.9, 0.0, -2.3]]
    s = [[0.5, 1.1, 1.4], [1.7, 0.7, 1.5]]
    res2 = sr.crps_mixnorm(obs, m, s, backend=backend)
    assert np.allclose(res1, res2)


def test_crps_negbinom(backend):
    if backend in ["jax", "torch", "tensorflow"]:
        pytest.skip("Not implemented in jax, torch or tensorflow backends")

    # test exceptions
    with pytest.raises(ValueError):
        sr.crps_negbinom(0.3, 7.0, 0.8, mu=7.3, backend=backend)

    # test correctness
    obs, n, prob = 2.0, 7.0, 0.8
    res = sr.crps_negbinom(obs, n, prob, backend=backend)
    expected = 0.3834322
    assert np.isclose(res, expected)

    obs, n, prob = 1.5, 2.0, 0.5
    res = sr.crps_negbinom(obs, n, prob, backend=backend)
    expected = 0.462963
    assert np.isclose(res, expected)

    obs, n, prob = -1.0, 17.0, 0.1
    res = sr.crps_negbinom(obs, n, prob, backend=backend)
    expected = 132.0942
    assert np.isclose(res, expected)

    obs, n, mu = 2.3, 11.0, 7.3
    res = sr.crps_negbinom(obs, n, mu=mu, backend=backend)
    expected = 3.149218
    assert np.isclose(res, expected)


def test_crps_normal(backend):
    obs = np.random.randn(N)
    mu = obs + np.random.randn(N) * 0.1
    sigma = abs(np.random.randn(N)) * 0.3

    # non-negative values
    res = sr.crps_normal(obs, mu, sigma, backend=backend)
    res = np.asarray(res)
    assert not np.any(np.isnan(res))
    assert not np.any(res < 0.0)

    # approx zero when perfect forecast
    mu = obs + np.random.randn(N) * 1e-6
    sigma = abs(np.random.randn(N)) * 1e-6
    res = sr.crps_normal(obs, mu, sigma, backend=backend)
    res = np.asarray(res)

    assert not np.any(np.isnan(res))
    assert not np.any(res - 0.0 > 0.0001)


def test_crps_2pnormal(backend):
    obs = np.random.randn(N)
    mu = obs + np.random.randn(N) * 0.1
    sigma1 = abs(np.random.randn(N)) * 0.3
    sigma2 = abs(np.random.randn(N)) * 0.2

    res = sr.crps_2pnormal(obs, sigma1, sigma2, mu, backend=backend)
    res = np.asarray(res)
    assert not np.any(np.isnan(res))
    assert not np.any(res < 0.0)


def test_crps_poisson(backend):
    obs, mean = 1.0, 3.0
    res = sr.crps_poisson(obs, mean, backend=backend)
    expected = 1.143447
    assert np.isclose(res, expected)

    obs, mean = 1.5, 2.3
    res = sr.crps_poisson(obs, mean, backend=backend)
    expected = 0.5001159
    assert np.isclose(res, expected)

    obs, mean = -1.0, 1.5
    res = sr.crps_poisson(obs, mean, backend=backend)
    expected = 1.840259
    assert np.isclose(res, expected)


def test_crps_t(backend):
    if backend in ["jax", "torch", "tensorflow"]:
        pytest.skip("Not implemented in jax, torch or tensorflow backends")

    obs, df, mu, sigma = 11.1, 5.2, 13.8, 2.3
    expected = 1.658226
    res = sr.crps_t(obs, df, mu, sigma, backend=backend)
    assert np.isclose(res, expected)

    obs, df = 0.7, 4.0
    expected = 0.4387929
    res = sr.crps_t(obs, df, backend=backend)
    assert np.isclose(res, expected)


def test_crps_uniform(backend):
    obs, min, max, lmass, umass = 0.3, -1.0, 2.1, 0.3, 0.1
    res = sr.crps_uniform(obs, min, max, lmass, umass, backend=backend)
    expected = 0.3960968
    assert np.isclose(res, expected)

    obs, min, max, lmass = -17.9, -15.2, -8.7, 0.2
    res = sr.crps_uniform(obs, min, max, lmass, backend=backend)
    expected = 4.086667
    assert np.isclose(res, expected)

    obs, min, max = 2.2, 0.1, 3.1
    res = sr.crps_uniform(obs, min, max, backend=backend)
    expected = 0.37
    assert np.isclose(res, expected)
