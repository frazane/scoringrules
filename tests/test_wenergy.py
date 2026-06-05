import numpy as np
import pytest

import scoringrules as sr
from scoringrules.backend import backends


ENSEMBLE_SIZE = 51
N = 100
N_VARS = 3


def test_owes_vs_es(backend):
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)

    res = sr.es_ensemble(obs, fct, backend=backend)
    resw = sr.owes_ensemble(
        obs,
        fct,
        lambda x: backends[backend].mean(x) * 0.0 + 1.0,
        backend=backend,
    )
    np.testing.assert_allclose(res, resw, atol=1e-6)


def test_twes_vs_es(backend):
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)

    res = sr.es_ensemble(obs, fct, backend=backend)
    resw = sr.twes_ensemble(obs, fct, lambda x: x, backend=backend)
    np.testing.assert_allclose(res, resw, rtol=1e-10)


def test_vres_vs_es(backend):
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)

    res = sr.es_ensemble(obs, fct, backend=backend)
    resw = sr.vres_ensemble(
        obs,
        fct,
        lambda x: backends[backend].mean(x) * 0.0 + 1.0,
        backend=backend,
    )
    np.testing.assert_allclose(res, resw, rtol=1e-6)


def test_owenergy_score_correctness(backend):
    fct = np.array(
        [[0.79546742, 0.4777960, 0.2164079], [0.02461368, 0.7584595, 0.3181810]]
    ).T
    obs = np.array([0.2743836, 0.8146400])

    def w_func(x):
        return backends[backend].all(x > 0.2)

    res = sr.owes_ensemble(obs, fct, w_func, backend=backend)
    np.testing.assert_allclose(res, 0.2274243, rtol=1e-6)

    def w_func(x):
        return backends[backend].all(x < 1.0)

    res = sr.owes_ensemble(obs, fct, w_func, backend=backend)
    np.testing.assert_allclose(res, 0.3345418, rtol=1e-6)


def test_twenergy_score_correctness(backend):
    fct = np.array(
        [[0.79546742, 0.4777960, 0.2164079], [0.02461368, 0.7584595, 0.3181810]]
    ).T
    obs = np.array([0.2743836, 0.8146400])

    def v_func(x):
        return np.maximum(x, 0.2)

    res = sr.twes_ensemble(obs, fct, v_func, backend=backend)
    np.testing.assert_allclose(res, 0.3116075, rtol=1e-6)

    def v_func(x):
        return np.minimum(x, 1)

    res = sr.twes_ensemble(obs, fct, v_func, backend=backend)
    np.testing.assert_allclose(res, 0.3345418, rtol=1e-6)


@pytest.mark.parametrize("score_fn", [sr.owes_ensemble, sr.vres_ensemble])
def test_weighted_energy_uniform_ens_w_equivalence(score_fn, backend):
    """Uniform ens_w must reproduce the unweighted score. This guards the
    weighted numba gufuncs (_owenergy_score_gufunc_w / _vrenergy_score_gufunc_w)
    against the trusted unweighted kernels."""
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)
    w_func = lambda x: backends[backend].mean(x) * 0.0 + 1.0  # noqa: E731
    ens_w = np.ones(fct.shape[:-1])
    res = np.asarray(score_fn(obs, fct, w_func, backend=backend))
    res_w = np.asarray(score_fn(obs, fct, w_func, ens_w=ens_w, backend=backend))
    np.testing.assert_allclose(res, res_w, atol=1e-6)
