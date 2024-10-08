import numpy as np
import pytest
from scoringrules import _logs

from .conftest import BACKENDS

ENSEMBLE_SIZE = 51
N = 100


@pytest.mark.parametrize("backend", BACKENDS)
def test_ensemble(backend):
    obs = np.random.randn(N)
    mu = obs + np.random.randn(N) * 0.1
    sigma = abs(np.random.randn(N))
    fct = np.random.randn(N, ENSEMBLE_SIZE) * sigma[..., None] + mu[..., None]

    res = _logs.logs_ensemble(obs, fct, axis=-1, backend=backend)
    assert res.shape == (N,)

    fct = fct.T
    res0 = _logs.logs_ensemble(obs, fct, axis=0, backend=backend)
    assert np.allclose(res, res0)

    obs, fct = 6.2, [4.2, 5.1, 6.1, 7.6, 8.3, 9.5]
    res = _logs.logs_ensemble(obs, fct, backend=backend)
    expected = 1.926475
    assert np.isclose(res, expected)

    fct = [-142.7, -160.3, -179.4, -184.5]
    res = _logs.logs_ensemble(obs, fct, backend=backend)
    expected = 55.25547
    assert np.isclose(res, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_clogs(backend):
    obs = np.random.randn(N)
    mu = obs + np.random.randn(N) * 0.1
    sigma = abs(np.random.randn(N))
    fct = np.random.randn(N, ENSEMBLE_SIZE) * sigma[..., None] + mu[..., None]

    res0 = _logs.logs_ensemble(obs, fct, axis=-1, backend=backend)
    res = _logs.clogs_ensemble(obs, fct, axis=-1, backend=backend)
    res_co = _logs.clogs_ensemble(obs, fct, axis=-1, cens=False, backend=backend)
    assert res.shape == (N,)
    assert res_co.shape == (N,)
    assert np.allclose(res, res0)
    assert np.allclose(res_co, res0)

    fct = fct.T
    res0 = _logs.clogs_ensemble(obs, fct, axis=0, backend=backend)
    assert np.allclose(res, res0)

    obs, fct = 6.2, [4.2, 5.1, 6.1, 7.6, 8.3, 9.5]
    a = 8.0
    res = _logs.clogs_ensemble(obs, fct, a=a, backend=backend)
    expected = 0.3929448
    assert np.isclose(res, expected)

    a = 5.0
    res = _logs.clogs_ensemble(obs, fct, a=a, cens=False, backend=backend)
    expected = 1.646852
    assert np.isclose(res, expected)

    fct = [-142.7, -160.3, -179.4, -184.5]
    b = -150.0
    res = _logs.clogs_ensemble(obs, fct, b=b, backend=backend)
    expected = 1.420427
    assert np.isclose(res, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_beta(backend):
    if backend == "torch":
        pytest.skip("Not implemented in torch backend")

    res = _logs.logs_beta(
        np.random.uniform(0, 1, (3, 3)),
        np.random.uniform(0, 3, (3, 3)),
        1.1,
        backend=backend,
    )
    assert res.shape == (3, 3)
    assert not np.any(np.isnan(res))

    # TODO: investigate why JAX results are different from other backends
    # (would fail test with smaller tolerance)
    obs, shape1, shape2, lower, upper = 0.3, 0.7, 1.1, 0.0, 1.0
    res = _logs.logs_beta(obs, shape1, shape2, lower, upper, backend=backend)
    expected = -0.04344567
    assert np.isclose(res, expected, atol=1e-4)

    obs, shape1, shape2, lower, upper = -3.0, 0.9, 2.1, -5.0, 4.0
    res = _logs.logs_beta(obs, shape1, shape2, lower, upper, backend=backend)
    expected = 1.74193
    assert np.isclose(res, expected)

    obs, shape1, shape2, lower, upper = -3.0, 0.9, 2.1, -2.0, 4.0
    res = _logs.logs_beta(obs, shape1, shape2, lower, upper, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_binomial(backend):
    # test correctness
    res = _logs.logs_binomial(8, 10, 0.9, backend=backend)
    expected = 1.641392
    assert np.isclose(res, expected)

    res = _logs.logs_binomial(-8, 10, 0.9, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    res = _logs.logs_binomial(18, 30, 0.5, backend=backend)
    expected = 2.518839
    assert np.isclose(res, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_exponential(backend):
    obs, rate = 0.3, 0.1
    res = _logs.logs_exponential(obs, rate, backend=backend)
    expected = 2.332585
    assert np.isclose(res, expected)

    obs, rate = -1.3, 2.4
    res = _logs.logs_exponential(obs, rate, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    obs, rate = 0.0, 0.9
    res = _logs.logs_exponential(obs, rate, backend=backend)
    expected = 0.1053605
    assert np.isclose(res, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_gamma(backend):
    obs, shape, rate = 0.2, 1.1, 0.7
    expected = 0.6434138

    res = _logs.logs_gamma(obs, shape, rate, backend=backend)
    assert np.isclose(res, expected)

    res = _logs.logs_gamma(obs, shape, scale=1 / rate, backend=backend)
    assert np.isclose(res, expected)

    with pytest.raises(ValueError):
        _logs.logs_gamma(obs, shape, rate, scale=1 / rate, backend=backend)
        return

    with pytest.raises(ValueError):
        _logs.logs_gamma(obs, shape, backend=backend)
        return


@pytest.mark.parametrize("backend", BACKENDS)
def test_gev(backend):
    obs, xi, mu, sigma = 0.3, 0.7, 0.0, 1.0
    res0 = _logs.logs_gev(obs, xi, backend=backend)
    expected = 1.22455
    assert np.isclose(res0, expected)

    res = _logs.logs_gev(obs, xi, mu, sigma, backend=backend)
    assert np.isclose(res, res0)

    obs, xi, mu, sigma = 0.3, -0.7, 0.0, 1.0
    res = _logs.logs_gev(obs, xi, mu, sigma, backend=backend)
    expected = 0.8151139
    assert np.isclose(res, expected)

    obs, xi, mu, sigma = 0.3, 0.0, 0.0, 1.0
    res = _logs.logs_gev(obs, xi, mu, sigma, backend=backend)
    expected = 1.040818
    assert np.isclose(res, expected)

    obs, xi, mu, sigma = -3.6, 0.7, 0.0, 1.0
    res = _logs.logs_gev(obs, xi, mu, sigma, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    obs, xi, mu, sigma = -3.6, 1.7, -4.0, 2.7
    res = _logs.logs_gev(obs, xi, mu, sigma, backend=backend)
    expected = 2.226233
    assert np.isclose(res, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_gpd(backend):
    obs, shape, location, scale = 0.8, 0.9, 0.0, 1.0
    res0 = _logs.logs_gpd(obs, shape, backend=backend)
    expected = 1.144907
    assert np.isclose(res0, expected)

    res = _logs.logs_gpd(obs, shape, location, scale, backend=backend)
    assert np.isclose(res0, res)

    obs = -0.8
    res = _logs.logs_gpd(obs, shape, location, scale, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    obs, shape = 0.8, -0.9
    res = _logs.logs_gpd(obs, shape, location, scale, backend=backend)
    expected = 0.1414406
    assert np.isclose(res, expected)

    shape, scale = 0.0, 2.1
    res = _logs.logs_gpd(obs, shape, location, scale, backend=backend)
    expected = 1.12289
    assert np.isclose(res, expected)

    obs, shape, location, scale = -17.4, 2.3, -21.0, 4.3
    res = _logs.logs_gpd(obs, shape, location, scale, backend=backend)
    expected = 2.998844
    assert np.isclose(res, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_hypergeometric(backend):
    res = _logs.logs_hypergeometric(5, 7, 13, 12)
    expected = 1.251525
    assert np.isclose(res, expected)

    res = _logs.logs_hypergeometric(5 * np.ones((2, 2)), 7, 13, 12, backend=backend)
    assert res.shape == (2, 2)

    res = _logs.logs_hypergeometric(5, 7 * np.ones((2, 2)), 13, 12, backend=backend)
    assert res.shape == (2, 2)


@pytest.mark.parametrize("backend", BACKENDS)
def test_logis(backend):
    obs, mu, sigma = 17.1, 13.8, 3.3
    res = _logs.logs_logistic(obs, mu, sigma, backend=backend)
    expected = 2.820446
    assert np.isclose(res, expected)

    obs, mu, sigma = 3.1, 4.0, 0.5
    res = _logs.logs_logistic(obs, mu, sigma, backend=backend)
    expected = 1.412808
    assert np.isclose(res, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_loglogistic(backend):
    obs, mulog, sigmalog = 3.0, 0.1, 0.9
    res = _logs.logs_loglogistic(obs, mulog, sigmalog, backend=backend)
    expected = 2.672729
    assert np.isclose(res, expected)

    obs, mulog, sigmalog = 0.0, 0.1, 0.9
    res = _logs.logs_loglogistic(obs, mulog, sigmalog, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    obs, mulog, sigmalog = 12.0, 12.1, 4.9
    res = _logs.logs_loglogistic(obs, mulog, sigmalog, backend=backend)
    expected = 6.299409
    assert np.isclose(res, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_lognormal(backend):
    obs, mulog, sigmalog = 3.0, 0.1, 0.9
    res = _logs.logs_lognormal(obs, mulog, sigmalog, backend=backend)
    expected = 2.527762
    assert np.isclose(res, expected)

    obs, mulog, sigmalog = 0.0, 0.1, 0.9
    res = _logs.logs_lognormal(obs, mulog, sigmalog, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    obs, mulog, sigmalog = 12.0, 12.1, 4.9
    res = _logs.logs_lognormal(obs, mulog, sigmalog, backend=backend)
    expected = 6.91832
    assert np.isclose(res, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_exponential2(backend):
    obs, location, scale = 8.3, 7.0, 1.2
    res = _logs.logs_exponential2(obs, location, scale, backend=backend)
    expected = 1.265655
    assert np.isclose(res, expected)

    obs, location, scale = -1.3, 2.4, 4.5
    res = _logs.logs_exponential2(obs, location, scale, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    obs, location, scale = 0.9, 0.9, 2.0
    res = _logs.logs_exponential2(obs, location, scale, backend=backend)
    expected = 0.6931472
    assert np.isclose(res, expected)

    res0 = _logs.logs_exponential(obs, 1 / scale, backend=backend)
    res = _logs.logs_exponential2(obs, 0.0, scale, backend=backend)
    assert np.isclose(res, res0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_2pexponential(backend):
    obs, scale1, scale2, location = 0.3, 0.1, 4.3, 0.0
    res = _logs.logs_2pexponential(obs, scale1, scale2, location, backend=backend)
    expected = 1.551372
    assert np.isclose(res, expected)

    obs, scale1, scale2, location = -20.8, 7.1, 2.0, -15.4
    res = _logs.logs_2pexponential(obs, scale1, scale2, location, backend=backend)
    expected = 2.968838
    assert np.isclose(res, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_laplace(backend):
    obs, location, scale = -3.0, 0.1, 0.9
    res = _logs.logs_laplace(obs, backend=backend)
    expected = 3.693147
    assert np.isclose(res, expected)

    res = _logs.logs_laplace(obs, location, backend=backend)
    expected = 3.793147
    assert np.isclose(res, expected)

    res = _logs.logs_laplace(obs, location, scale, backend=backend)
    expected = 4.032231
    assert np.isclose(res, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_loglaplace(backend):
    obs, locationlog, scalelog = 3.0, 0.1, 0.9
    res = _logs.logs_loglaplace(obs, locationlog, scalelog, backend=backend)
    expected = 2.795968
    assert np.isclose(res, expected)

    obs, locationlog, scalelog = 0.0, 0.1, 0.9
    res = _logs.logs_loglaplace(obs, locationlog, scalelog, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    obs, locationlog, scalelog = 12.0, 12.1, 4.9
    res = _logs.logs_loglaplace(obs, locationlog, scalelog, backend=backend)
    expected = 6.729553
    assert np.isclose(res, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_mixnorm(backend):
    obs, m, s, w = 0.3, [0.0, -2.9, 0.9], [0.5, 1.4, 0.7], [1 / 3, 1 / 3, 1 / 3]
    res = _logs.logs_mixnorm(obs, m, s, w, backend=backend)
    expected = 1.019742
    assert np.isclose(res, expected)

    res0 = _logs.logs_mixnorm(obs, m, s, backend=backend)
    assert np.isclose(res, res0)

    w = [0.3, 0.1, 0.6]
    res = _logs.logs_mixnorm(obs, m, s, w, backend=backend)
    expected = 0.8235977
    assert np.isclose(res, expected)

    obs = [-1.6, 0.3]
    m = [[0.0, -2.9], [0.6, 0.0], [-1.1, -2.3]]
    s = [[0.5, 1.7], [1.1, 0.7], [1.4, 1.5]]
    res1 = _logs.logs_mixnorm(obs, m, s, axis=0, backend=backend)

    m = [[0.0, 0.6, -1.1], [-2.9, 0.0, -2.3]]
    s = [[0.5, 1.1, 1.4], [1.7, 0.7, 1.5]]
    res2 = _logs.logs_mixnorm(obs, m, s, backend=backend)
    assert np.allclose(res1, res2)


@pytest.mark.parametrize("backend", BACKENDS)
def test_negbinom(backend):
    if backend in ["jax", "torch"]:
        pytest.skip("Not implemented in jax or torch backends")

    # TODO: investigate why JAX and torch results are different from other backends
    # (they fail the test)
    obs, n, prob = 2.0, 7.0, 0.8
    res = _logs.logs_negbinom(obs, n, prob, backend=backend)
    expected = 1.448676
    assert np.isclose(res, expected)

    obs, n, prob = 1.5, 2.0, 0.5
    res = _logs.logs_negbinom(obs, n, prob, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    obs, n, prob = -1.0, 17.0, 0.1
    res = _logs.logs_negbinom(obs, n, prob, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    obs, n, mu = 2.0, 11.0, 7.3
    res = _logs.logs_negbinom(obs, n, mu=mu, backend=backend)
    expected = 3.247462
    assert np.isclose(res, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_normal(backend):
    res = _logs.logs_normal(0.0, 0.1, 0.1, backend=backend)
    assert np.isclose(res, -0.8836466, rtol=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_2pnormal(backend):
    obs, scale1, scale2, location = 29.1, 4.6, 1.3, 27.9
    res = _logs.logs_2pnormal(obs, scale1, scale2, location, backend=backend)
    expected = 2.426779
    assert np.isclose(res, expected)

    obs, scale1, scale2, location = -2.2, 1.6, 3.3, -1.9
    res = _logs.logs_2pnormal(obs, scale1, scale2, location, backend=backend)
    expected = 1.832605
    assert np.isclose(res, expected)

    obs, scale, location = 1.5, 4.5, 5.4
    res0 = _logs.logs_normal(obs, location, scale, backend=backend)
    res = _logs.logs_2pnormal(obs, scale, scale, location, backend=backend)
    assert np.isclose(res, res0)
    obs, mu, sigma = 17.1, 13.8, 3.3
    res = _logs.logs_normal(obs, mu, sigma, backend=backend)
    expected = 2.612861
    assert np.isclose(res, expected)

    obs, mu, sigma = 3.1, 4.0, 0.5
    res = _logs.logs_normal(obs, mu, sigma, backend=backend)
    expected = 1.845791
    assert np.isclose(res, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_poisson(backend):
    obs, mean = 1.0, 3.0
    res = _logs.logs_poisson(obs, mean, backend=backend)
    expected = 1.901388
    assert np.isclose(res, expected)

    obs, mean = 1.5, 2.3
    res = _logs.logs_poisson(obs, mean, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    obs, mean = -1.0, 1.5
    res = _logs.logs_poisson(obs, mean, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_t(backend):
    if backend in ["jax", "torch", "tensorflow"]:
        pytest.skip("Not implemented in jax, torch or tensorflow backends")

    obs, df, mu, sigma = 11.1, 5.2, 13.8, 2.3
    res = _logs.logs_t(obs, df, mu, sigma, backend=backend)
    expected = 2.528398
    assert np.isclose(res, expected)

    obs, df = 0.7, 4.0
    res = _logs.logs_t(obs, df, backend=backend)
    expected = 1.269725
    assert np.isclose(res, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_tlogis(backend):
    obs, location, scale, lower, upper = 4.9, 3.5, 2.3, 0.0, 20.0
    res = _logs.logs_tlogistic(obs, location, scale, lower, upper, backend=backend)
    expected = 2.11202
    assert np.isclose(res, expected)

    # aligns with logs_logistic
    res0 = _logs.logs_logistic(obs, location, scale, backend=backend)
    res = _logs.logs_tlogistic(obs, location, scale, backend=backend)
    assert np.isclose(res, res0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_tnormal(backend):
    obs, location, scale, lower, upper = 4.2, 2.9, 2.2, 1.5, 17.3
    res = _logs.logs_tnormal(obs, location, scale, lower, upper, backend=backend)
    expected = 1.577806
    assert np.isclose(res, expected)

    obs, location, scale, lower, upper = -1.0, 2.9, 2.2, 1.5, 17.3
    res = _logs.logs_tnormal(obs, location, scale, lower, upper, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    # aligns with logs_normal
    res0 = _logs.logs_normal(obs, location, scale, backend=backend)
    res = _logs.logs_tnormal(obs, location, scale, backend=backend)
    assert np.isclose(res, res0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_tt(backend):
    if backend in ["jax", "torch", "tensorflow"]:
        pytest.skip("Not implemented in jax, torch or tensorflow backends")

    obs, df, location, scale, lower, upper = 1.9, 2.9, 3.1, 4.2, 1.5, 17.3
    res = _logs.logs_tt(obs, df, location, scale, lower, upper, backend=backend)
    expected = 2.002856
    assert np.isclose(res, expected)

    obs, df, location, scale, lower, upper = -1.0, 2.9, 3.1, 4.2, 1.5, 17.3
    res = _logs.logs_tt(obs, df, location, scale, lower, upper, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    # aligns with logs_t
    res0 = _logs.logs_t(obs, df, location, scale, backend=backend)
    res = _logs.logs_tt(obs, df, location, scale, backend=backend)
    assert np.isclose(res, res0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_uniform(backend):
    obs, min, max = 0.3, -1.0, 2.1
    res = _logs.logs_uniform(obs, min, max, backend=backend)
    expected = 1.131402
    assert np.isclose(res, expected)

    obs, min, max = -17.9, -15.2, -8.7
    res = _logs.logs_uniform(obs, min, max, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    obs, min, max = 0.1, 0.1, 3.1
    res = _logs.logs_uniform(obs, min, max, backend=backend)
    expected = 1.098612
    assert np.isclose(res, expected)
