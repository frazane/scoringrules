import numpy as np
import pytest

import scoringrules as sr
from .conftest import assert_inferred

ENSEMBLE_SIZE = 51
N = 100


def assert_close(result, expected, backend, *, rtol=1e-5, atol=1e-8):
    """Compare a (possibly framework-native) result to an expected value.

    jax runs in float32 in the test environment (``jax_enable_x64`` is off), so its
    tolerances are loosened. torch runs in float64 and stays strict.
    """
    arr = np.asarray(result)
    if backend == "jax":
        rtol = max(rtol, 1e-4)
        atol = max(atol, 1e-5)
    assert np.allclose(
        arr, np.asarray(expected), rtol=rtol, atol=atol
    ), f"[{backend}] {arr} != {expected}"


@pytest.fixture()
def fa(to_backend):
    """Convert to a float array of the backend under test."""
    return lambda x: to_backend(np.asarray(x, dtype=float))


def test_ensemble(backend, fa, backend_kwargs):
    obs = np.random.randn(N)
    mu = obs + np.random.randn(N) * 0.1
    sigma = abs(np.random.randn(N))
    fct = np.random.randn(N, ENSEMBLE_SIZE) * sigma[..., None] + mu[..., None]

    res = sr.logs_ensemble(fa(obs), fa(fct), m_axis=-1, **backend_kwargs)
    assert_inferred(res, backend)
    assert res.shape == (N,)

    res0 = sr.logs_ensemble(fa(obs), fa(fct.T), m_axis=0, **backend_kwargs)
    assert_inferred(res0, backend)
    assert_close(res, np.asarray(res0), backend)

    # a given bandwidth, one per forecast
    bw = np.full(N, 0.3)
    res_bw = sr.logs_ensemble(fa(obs), fa(fct), bw=fa(bw), **backend_kwargs)
    assert_inferred(res_bw, backend)
    expected = sr.logs_mixnorm(obs, fct, np.full(fct.shape, 0.3))
    assert_close(res_bw, expected, backend)

    obs, fct = 6.2, [4.2, 5.1, 6.1, 7.6, 8.3, 9.5]
    res = sr.logs_ensemble(fa(obs), fa(fct), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 1.926475, backend)

    fct = [-142.7, -160.3, -179.4, -184.5]
    res = sr.logs_ensemble(fa(obs), fa(fct), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 55.25547, backend)


def test_clogs(backend, fa, backend_kwargs):
    obs = np.random.randn(N)
    mu = obs + np.random.randn(N) * 0.1
    sigma = abs(np.random.randn(N))
    fct = np.random.randn(N, ENSEMBLE_SIZE) * sigma[..., None] + mu[..., None]

    res0 = sr.logs_ensemble(fa(obs), fa(fct), m_axis=-1, **backend_kwargs)
    res = sr.clogs_ensemble(fa(obs), fa(fct), m_axis=-1, **backend_kwargs)
    res_co = sr.clogs_ensemble(
        fa(obs), fa(fct), m_axis=-1, cens=False, **backend_kwargs
    )
    assert_inferred(res, backend)
    assert_inferred(res_co, backend)
    assert res.shape == (N,)
    assert res_co.shape == (N,)
    assert_close(res, np.asarray(res0), backend, atol=1e-5)
    assert_close(res_co, np.asarray(res0), backend, atol=1e-5)

    res1 = sr.clogs_ensemble(fa(obs), fa(fct.T), m_axis=0, **backend_kwargs)
    assert_inferred(res1, backend)
    assert_close(res, np.asarray(res1), backend, atol=1e-5)

    obs, fct = 6.2, [4.2, 5.1, 6.1, 7.6, 8.3, 9.5]
    res = sr.clogs_ensemble(fa(obs), fa(fct), a=fa(8.0), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 0.3929448, backend)

    res = sr.clogs_ensemble(fa(obs), fa(fct), a=fa(5.0), cens=False, **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 1.646852, backend)

    fct = [-142.7, -160.3, -179.4, -184.5]
    res = sr.clogs_ensemble(fa(obs), fa(fct), b=fa(-150.0), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 1.420427, backend)


def test_beta(backend, fa, backend_kwargs):
    res = sr.logs_beta(
        fa(np.random.uniform(0, 1, (3, 3))),
        fa(np.random.uniform(0, 3, (3, 3))),
        fa(1.1),
        **backend_kwargs,
    )
    assert_inferred(res, backend)
    assert res.shape == (3, 3)
    assert not np.any(np.isnan(np.asarray(res)))

    # jax runs in float32, hence the loose tolerance on this case
    res = sr.logs_beta(fa(0.3), fa(0.7), fa(1.1), fa(0.0), fa(1.0), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, -0.04344567, backend, atol=1e-4)

    res = sr.logs_beta(fa(-3.0), fa(0.9), fa(2.1), fa(-5.0), fa(4.0), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 1.74193, backend)

    res = sr.logs_beta(fa(-3.0), fa(0.9), fa(2.1), fa(-2.0), fa(4.0), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, float("inf"), backend)


def test_binomial(backend, fa, backend_kwargs):
    res = sr.logs_binomial(fa(8), fa(10), fa(0.9), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 1.641392, backend)

    res = sr.logs_binomial(fa(-8), fa(10), fa(0.9), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, float("inf"), backend)

    res = sr.logs_binomial(fa(18), fa(30), fa(0.5), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 2.518839, backend)

    # large n: comb(171, 50) overflows a factorial-based comb (171! > 1e308).
    # -scipy.stats.binom.logpmf(50, 171, 0.4)
    res = sr.logs_binomial(fa(50), fa(171), fa(0.4), **backend_kwargs)
    assert_inferred(res, backend)
    # comb(171, 50) ~ 1e43 does not fit jax's float32 (x64 off), so only the
    # numeric comparison is skipped on jax.
    if backend != "jax":
        assert_close(res, 6.995656745750523, backend)


def test_exponential(backend, fa, backend_kwargs):
    res = sr.logs_exponential(fa(0.3), fa(0.1), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 2.332585, backend)

    res = sr.logs_exponential(fa(-1.3), fa(2.4), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, float("inf"), backend)

    res = sr.logs_exponential(fa(0.0), fa(0.9), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 0.1053605, backend)


def test_gamma(backend, fa, backend_kwargs):
    obs, shape, rate = 0.2, 1.1, 0.7
    expected = 0.6434138

    res = sr.logs_gamma(fa(obs), fa(shape), fa(rate), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, expected, backend)

    res = sr.logs_gamma(fa(obs), fa(shape), scale=fa(1 / rate), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, expected, backend)

    with pytest.raises(ValueError):
        sr.logs_gamma(
            fa(obs), fa(shape), fa(rate), scale=fa(1 / rate), **backend_kwargs
        )

    with pytest.raises(ValueError):
        sr.logs_gamma(fa(obs), fa(shape), **backend_kwargs)


def test_gev(backend, fa, backend_kwargs):
    res0 = sr.logs_gev(fa(0.3), fa(0.7), **backend_kwargs)
    assert_inferred(res0, backend)
    assert_close(res0, 1.22455, backend)

    res = sr.logs_gev(fa(0.3), fa(0.7), fa(0.0), fa(1.0), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, np.asarray(res0), backend)

    res = sr.logs_gev(fa(0.3), fa(-0.7), fa(0.0), fa(1.0), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 0.8151139, backend)

    res = sr.logs_gev(fa(0.3), fa(0.0), fa(0.0), fa(1.0), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 1.040818, backend)

    res = sr.logs_gev(fa(-3.6), fa(0.7), fa(0.0), fa(1.0), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, float("inf"), backend)

    res = sr.logs_gev(fa(-3.6), fa(1.7), fa(-4.0), fa(2.7), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 2.226233, backend)


def test_gpd(backend, fa, backend_kwargs):
    res0 = sr.logs_gpd(fa(0.8), fa(0.9), **backend_kwargs)
    assert_inferred(res0, backend)
    assert_close(res0, 1.144907, backend)

    res = sr.logs_gpd(fa(0.8), fa(0.9), fa(0.0), fa(1.0), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, np.asarray(res0), backend)

    res = sr.logs_gpd(fa(-0.8), fa(0.9), fa(0.0), fa(1.0), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, float("inf"), backend)

    res = sr.logs_gpd(fa(0.8), fa(-0.9), fa(0.0), fa(1.0), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 0.1414406, backend)

    res = sr.logs_gpd(fa(0.8), fa(0.0), fa(0.0), fa(2.1), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 1.12289, backend)

    res = sr.logs_gpd(fa(-17.4), fa(2.3), fa(-21.0), fa(4.3), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 2.998844, backend)


def test_hypergeometric(backend, fa, backend_kwargs):
    res = sr.logs_hypergeometric(fa(5), fa(7), fa(13), fa(12), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 1.251525, backend)

    # large population: -scipy.stats.hypergeom.logpmf(5, 171, 100, 20)
    res = sr.logs_hypergeometric(fa(5), fa(100), fa(71), fa(20), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 6.755243793500753, backend)

    res = sr.logs_hypergeometric(
        fa(5 * np.ones((2, 2))), fa(7), fa(13), fa(12), **backend_kwargs
    )
    assert_inferred(res, backend)
    assert res.shape == (2, 2)

    res = sr.logs_hypergeometric(
        fa(5), fa(7 * np.ones((2, 2))), fa(13), fa(12), **backend_kwargs
    )
    assert_inferred(res, backend)
    assert res.shape == (2, 2)


def test_logis(backend, fa, backend_kwargs):
    res = sr.logs_logistic(fa(17.1), fa(13.8), fa(3.3), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 2.820446, backend)

    res = sr.logs_logistic(fa(3.1), fa(4.0), fa(0.5), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 1.412808, backend)


def test_loglogistic(backend, fa, backend_kwargs):
    res = sr.logs_loglogistic(fa(3.0), fa(0.1), fa(0.9), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 2.672729, backend)

    res = sr.logs_loglogistic(fa(0.0), fa(0.1), fa(0.9), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, float("inf"), backend)

    res = sr.logs_loglogistic(fa(12.0), fa(12.1), fa(4.9), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 6.299409, backend)


def test_lognormal(backend, fa, backend_kwargs):
    res = sr.logs_lognormal(fa(3.0), fa(0.1), fa(0.9), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 2.527762, backend)

    res = sr.logs_lognormal(fa(0.0), fa(0.1), fa(0.9), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, float("inf"), backend)

    res = sr.logs_lognormal(fa(12.0), fa(12.1), fa(4.9), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 6.91832, backend)


def test_exponential2(backend, fa, backend_kwargs):
    res = sr.logs_exponential2(fa(8.3), fa(7.0), fa(1.2), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 1.265655, backend)

    res = sr.logs_exponential2(fa(-1.3), fa(2.4), fa(4.5), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, float("inf"), backend)

    res = sr.logs_exponential2(fa(0.9), fa(0.9), fa(2.0), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 0.6931472, backend)

    res0 = sr.logs_exponential(fa(0.9), fa(1 / 2.0), **backend_kwargs)
    res = sr.logs_exponential2(fa(0.9), fa(0.0), fa(2.0), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, np.asarray(res0), backend)


def test_exponential2_does_not_modify_obs(backend, fa, backend_kwargs):
    # #153: the location shift used to be applied in place to the caller's obs
    obs = fa([1.0, 2.0])
    sr.logs_exponential2(obs, fa(0.5), fa(1.0), **backend_kwargs)
    np.testing.assert_array_equal(np.asarray(obs), [1.0, 2.0])


def test_2pexponential(backend, fa, backend_kwargs):
    res = sr.logs_2pexponential(fa(0.3), fa(0.1), fa(4.3), fa(0.0), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 1.551372, backend)

    res = sr.logs_2pexponential(
        fa(-20.8), fa(7.1), fa(2.0), fa(-15.4), **backend_kwargs
    )
    assert_inferred(res, backend)
    assert_close(res, 2.968838, backend)


def test_laplace(backend, fa, backend_kwargs):
    res = sr.logs_laplace(fa(-3.0), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 3.693147, backend)

    res = sr.logs_laplace(fa(-3.0), fa(0.1), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 3.793147, backend)

    res = sr.logs_laplace(fa(-3.0), fa(0.1), fa(0.9), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 4.032231, backend)


def test_loglaplace(backend, fa, backend_kwargs):
    res = sr.logs_loglaplace(fa(3.0), fa(0.1), fa(0.9), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 2.795968, backend)

    res = sr.logs_loglaplace(fa(0.0), fa(0.1), fa(0.9), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, float("inf"), backend)

    res = sr.logs_loglaplace(fa(12.0), fa(12.1), fa(4.9), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 6.729553, backend)


def test_mixnorm(backend, fa, backend_kwargs):
    obs, m, s, w = 0.3, [0.0, -2.9, 0.9], [0.5, 1.4, 0.7], [1 / 3, 1 / 3, 1 / 3]
    res = sr.logs_mixnorm(fa(obs), fa(m), fa(s), fa(w), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 1.019742, backend)

    res0 = sr.logs_mixnorm(fa(obs), fa(m), fa(s), **backend_kwargs)
    assert_inferred(res0, backend)
    assert_close(res, np.asarray(res0), backend)

    w = [0.3, 0.1, 0.6]
    res = sr.logs_mixnorm(fa(obs), fa(m), fa(s), fa(w), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 0.8235977, backend)

    obs = [-1.6, 0.3]
    m = [[0.0, -2.9], [0.6, 0.0], [-1.1, -2.3]]
    s = [[0.5, 1.7], [1.1, 0.7], [1.4, 1.5]]
    res1 = sr.logs_mixnorm(fa(obs), fa(m), fa(s), mc_axis=0, **backend_kwargs)
    assert_inferred(res1, backend)

    m = [[0.0, 0.6, -1.1], [-2.9, 0.0, -2.3]]
    s = [[0.5, 1.1, 1.4], [1.7, 0.7, 1.5]]
    res2 = sr.logs_mixnorm(fa(obs), fa(m), fa(s), **backend_kwargs)
    assert_inferred(res2, backend)
    assert_close(res1, np.asarray(res2), backend)


def test_negbinom(backend, fa, backend_kwargs):
    with pytest.raises(ValueError):
        sr.logs_negbinom(fa(1), fa(1.1), fa(0.1), mu=fa(0.1), **backend_kwargs)

    res = sr.logs_negbinom(fa(2.0), fa(7.0), fa(0.8), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 1.448676, backend)

    res = sr.logs_negbinom(fa(1.5), fa(2.0), fa(0.5), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, float("inf"), backend)

    res = sr.logs_negbinom(fa(-1.0), fa(17.0), fa(0.1), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, float("inf"), backend)

    res = sr.logs_negbinom(fa(2.0), fa(11.0), mu=fa(7.3), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 3.247462, backend)

    # a real-valued size: scipy's comb, not a rounded integer comb
    res = sr.logs_negbinom(fa(3.0), fa(2.5), fa(0.4), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 1.9418321, backend)


def test_normal(backend, fa, backend_kwargs):
    res = sr.logs_normal(fa(0.0), fa(0.1), fa(0.1), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, -0.8836466, backend)

    res = sr.logs_normal(fa(17.1), fa(13.8), fa(3.3), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 2.612861, backend)

    res = sr.logs_normal(fa(3.1), fa(4.0), fa(0.5), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 1.845791, backend)

    res = sr.logs_normal(fa(3.1), fa(4.0), fa(0.5), negative=False, **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, -1.845791, backend)


def test_2pnormal(backend, fa, backend_kwargs):
    res = sr.logs_2pnormal(fa(29.1), fa(4.6), fa(1.3), fa(27.9), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 2.426779, backend)

    res = sr.logs_2pnormal(fa(-2.2), fa(1.6), fa(3.3), fa(-1.9), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 1.832605, backend)

    obs, scale, location = 1.5, 4.5, 5.4
    res0 = sr.logs_normal(fa(obs), fa(location), fa(scale), **backend_kwargs)
    res = sr.logs_2pnormal(
        fa(obs), fa(scale), fa(scale), fa(location), **backend_kwargs
    )
    assert_inferred(res, backend)
    assert_close(res, np.asarray(res0), backend)


def test_poisson(backend, fa, backend_kwargs):
    res = sr.logs_poisson(fa(1.0), fa(3.0), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 1.901388, backend)

    res = sr.logs_poisson(fa(1.5), fa(2.3), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, float("inf"), backend)

    res = sr.logs_poisson(fa(-1.0), fa(1.5), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, float("inf"), backend)


def test_t(backend, fa, backend_kwargs):
    res = sr.logs_t(fa(11.1), fa(5.2), fa(13.8), fa(2.3), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 2.528398, backend)

    res = sr.logs_t(fa(0.7), fa(4.0), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 1.269725, backend)


def test_tlogis(backend, fa, backend_kwargs):
    res = sr.logs_tlogistic(
        fa(4.9), fa(3.5), fa(2.3), fa(0.0), fa(20.0), **backend_kwargs
    )
    assert_inferred(res, backend)
    assert_close(res, 2.11202, backend)

    # aligns with logs_logistic
    res0 = sr.logs_logistic(fa(4.9), fa(3.5), fa(2.3), **backend_kwargs)
    res = sr.logs_tlogistic(fa(4.9), fa(3.5), fa(2.3), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, np.asarray(res0), backend)


def test_tnormal(backend, fa, backend_kwargs):
    res = sr.logs_tnormal(
        fa(4.2), fa(2.9), fa(2.2), fa(1.5), fa(17.3), **backend_kwargs
    )
    assert_inferred(res, backend)
    assert_close(res, 1.577806, backend)

    res = sr.logs_tnormal(
        fa(-1.0), fa(2.9), fa(2.2), fa(1.5), fa(17.3), **backend_kwargs
    )
    assert_inferred(res, backend)
    assert_close(res, float("inf"), backend)

    # aligns with logs_normal
    res0 = sr.logs_normal(fa(-1.0), fa(2.9), fa(2.2), **backend_kwargs)
    res = sr.logs_tnormal(fa(-1.0), fa(2.9), fa(2.2), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, np.asarray(res0), backend)


def test_tt(backend, fa, backend_kwargs):
    if backend == "torch":
        with pytest.raises(NotImplementedError):
            sr.logs_tt(fa(1.9), fa(2.9), fa(3.1), fa(4.2), fa(1.5), fa(17.3))
        pytest.skip(
            "betainc is BLOCKED on torch (docs/special_functions_audit.md, "
            "betainc row)"
        )

    res = sr.logs_tt(
        fa(1.9), fa(2.9), fa(3.1), fa(4.2), fa(1.5), fa(17.3), **backend_kwargs
    )
    assert_inferred(res, backend)
    assert_close(res, 2.002856, backend)

    res = sr.logs_tt(
        fa(-1.0), fa(2.9), fa(3.1), fa(4.2), fa(1.5), fa(17.3), **backend_kwargs
    )
    assert_inferred(res, backend)
    assert_close(res, float("inf"), backend)

    # aligns with logs_t
    res0 = sr.logs_t(fa(-1.0), fa(2.9), fa(3.1), fa(4.2), **backend_kwargs)
    res = sr.logs_tt(fa(-1.0), fa(2.9), fa(3.1), fa(4.2), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, np.asarray(res0), backend)


def test_uniform(backend, fa, backend_kwargs):
    res = sr.logs_uniform(fa(0.3), fa(-1.0), fa(2.1), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 1.131402, backend)

    res = sr.logs_uniform(fa(-17.9), fa(-15.2), fa(-8.7), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, float("inf"), backend)

    res = sr.logs_uniform(fa(0.1), fa(0.1), fa(3.1), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 1.098612, backend)


# One case per distribution: (function name, args). Used for the float32 test.
FLOAT32_CASES = [
    ("logs_ensemble", ([0.2, 1.1], [[0.1, -0.5, 0.9, 1.3], [0.8, 1.6, 0.2, 1.0]])),
    (
        "clogs_ensemble",
        ([0.2, 1.1], [[0.1, -0.5, 0.9, 1.3], [0.8, 1.6, 0.2, 1.0]], -1.0, 1.0),
    ),
    ("logs_beta", ([0.3, 0.6], 0.7, 1.1, 0.0, 1.0)),
    ("logs_binomial", ([8.0, 3.0], 10.0, 0.6)),
    ("logs_binomial", ([20.0, 14.0], 35.0, 0.4)),
    ("logs_exponential", ([0.3, 1.2], 0.9)),
    ("logs_exponential2", ([0.3, 1.2], 0.1, 2.0)),
    ("logs_2pexponential", ([0.3, -1.2], 0.5, 2.0, 0.1)),
    ("logs_gamma", ([0.2, 1.5], 1.1, 0.7)),
    ("logs_gev", ([0.3, 1.5], 0.3, 0.1, 1.2)),
    ("logs_gpd", ([0.3, 1.5], 0.3, 0.1, 1.2)),
    ("logs_hypergeometric", ([5.0, 6.0], 7.0, 13.0, 12.0)),
    ("logs_laplace", ([0.3, -1.5], 0.1, 1.2)),
    ("logs_loglaplace", ([0.3, 1.5], 0.1, 1.2)),
    ("logs_logistic", ([0.3, -1.5], 0.1, 1.2)),
    ("logs_loglogistic", ([0.3, 1.5], 0.1, 1.2)),
    ("logs_lognormal", ([0.3, 1.5], 0.1, 1.2)),
    ("logs_mixnorm", ([0.3, 1.5], [0.0, -2.9, 0.9], [0.5, 1.4, 0.7])),
    ("logs_negbinom", ([2.0, 4.0], 7.0, 0.8)),
    ("logs_normal", ([0.3, -1.5], 0.1, 1.2)),
    ("logs_2pnormal", ([0.3, -1.5], 0.5, 2.0, 0.1)),
    ("logs_poisson", ([1.0, 4.0], 3.0)),
    ("logs_t", ([0.3, -1.5], 4.0, 0.1, 1.2)),
    ("logs_tlogistic", ([0.3, 1.5], 0.1, 1.2, -1.0, 2.0)),
    ("logs_tnormal", ([0.3, 1.5], 0.1, 1.2, -1.0, 2.0)),
    ("logs_tt", ([0.3, 1.5], 4.0, 0.1, 1.2, -1.0, 2.0)),
    ("logs_uniform", ([0.3, 1.5], -1.0, 2.0)),
]


FLOAT32_IDS = [f"{c[0]}-{i}" for i, c in enumerate(FLOAT32_CASES)]


@pytest.mark.parametrize("name, args", FLOAT32_CASES, ids=FLOAT32_IDS)
def test_float32(name, args, backend, to_backend_float32, backend_kwargs):
    if backend == "torch" and name == "logs_tt":
        pytest.skip(
            "betainc is BLOCKED on torch (docs/special_functions_audit.md, "
            "betainc row)"
        )
    fn = getattr(sr, name)
    res = fn(*map(to_backend_float32, args), **backend_kwargs)
    assert_inferred(res, backend)
    assert str(res.dtype).endswith("float32"), f"{name} returned {res.dtype}"
    expected = np.asarray(fn(*(np.asarray(a, dtype=np.float64) for a in args)))
    np.testing.assert_allclose(np.asarray(res), expected, rtol=1e-4, atol=1e-5)


def test_exponential2_float32_python_location(backend, to_backend_float32):
    # the #153 fix (obs - location, not in place) must keep float32 obs in float32
    obs = to_backend_float32([0.3, 1.2])
    res = sr.logs_exponential2(obs, -1.0, to_backend_float32(2.0))
    assert_inferred(res, backend)
    assert str(res.dtype).endswith("float32"), res.dtype


LOCATION_SCALE = {
    "logs_exponential2",
    "logs_2pexponential",
    "logs_gev",
    "logs_gpd",
    "logs_laplace",
    "logs_loglaplace",
    "logs_logistic",
    "logs_loglogistic",
    "logs_lognormal",
    "logs_normal",
    "logs_2pnormal",
    "logs_t",
    "logs_tlogistic",
    "logs_tnormal",
    "logs_tt",
    "logs_uniform",
}
PYFLOAT_CASES = [c for c in FLOAT32_CASES if c[0] in LOCATION_SCALE]


@pytest.mark.parametrize("name, args", PYFLOAT_CASES, ids=[c[0] for c in PYFLOAT_CASES])
def test_float32_python_float_params(
    name, args, backend, to_backend_float32, backend_kwargs, request
):
    """float32 obs with Python-float parameters must stay float32."""
    if backend == "torch" and name == "logs_tt":
        pytest.skip(
            "betainc is BLOCKED on torch (docs/special_functions_audit.md, "
            "betainc row)"
        )
    if backend in ("numpy", "numba"):
        request.applymarker(
            pytest.mark.xfail(
                strict=True,
                reason="xp.asarray turns a Python float into a 0-d float64 array, "
                "which upcasts float32 (as on main); fixed by adopting "
                "utils_xp.asarray_matching from PR 2",
            )
        )
    fn = getattr(sr, name)
    res = fn(to_backend_float32(args[0]), *args[1:], **backend_kwargs)
    assert_inferred(res, backend)
    assert str(res.dtype).endswith("float32"), f"{name} returned {res.dtype}"


def test_logs_legacy_backend_string_deprecated():
    with pytest.warns(DeprecationWarning):
        sr.logs_normal(0.0, 0.1, 0.1, backend="numpy")


def test_logs_numba_backend_means_infer():
    torch = pytest.importorskip("torch")
    obs = torch.tensor([0.2, 1.0], dtype=torch.float64)
    res = sr.logs_normal(obs, 0.1, 1.2, backend="numba")
    assert_inferred(res, "torch")


# (function name, args, index of the argument to differentiate). The truncated
# scores are the #144 regression: they used to evaluate the pdf without the
# backend, i.e. through numpy, which breaks autograd.
TORCH_GRAD_CASES = [
    ("logs_tnormal", (0.3, 0.1, 1.2, -1.0, 2.0), 1),
    ("logs_tlogistic", (0.3, 0.1, 1.2, -1.0, 2.0), 1),
    ("logs_normal", (0.3, 0.1, 1.2), 2),
    ("logs_ensemble", ([0.2, 1.1], [[0.1, -0.5, 0.9, 1.3], [0.8, 1.6, 0.2, 1.0]]), 1),
    ("logs_mixnorm", (0.3, [0.0, -2.9, 0.9], [0.5, 1.4, 0.7]), 1),
    ("logs_gamma", (0.2, 1.1, 0.7), 1),
    ("logs_gev", (0.3, 0.3, 0.1, 1.2), 1),
    ("logs_beta", (0.3, 0.7, 1.1), 1),
    ("logs_t", (0.3, 4.0, 0.1, 1.2), 1),
]


@pytest.mark.parametrize(
    "name, args, i", TORCH_GRAD_CASES, ids=[c[0] for c in TORCH_GRAD_CASES]
)
def test_grad_torch(name, args, i):
    torch = pytest.importorskip("torch")
    args = [torch.tensor(a, dtype=torch.float64) for a in args]
    args[i].requires_grad_(True)
    res = getattr(sr, name)(*args)
    res.sum().backward()
    assert torch.isfinite(args[i].grad).all()


def test_tt_grad_jax():
    # #144 for logs_tt: betainc is BLOCKED on torch (audit, betainc row), so the
    # regression is checked through jax.grad instead.
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    def f(scale):
        obs = jnp.asarray([0.3, 1.5])
        return sr.logs_tt(obs, 4.0, 0.1, scale, -1.0, 2.0).sum()

    g = jax.grad(f)(jnp.asarray(1.2))
    assert jnp.isfinite(g)
