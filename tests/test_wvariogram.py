import numpy as np
import pytest


import scoringrules as sr
from scoringrules.backend import backends

from ._parity import MV_CASES, mark_known_mv_failures, mv_ensemble_inputs, mv_w_func


ENSEMBLE_SIZE = 51
N = 100
N_VARS = 3


def test_owvs_vs_vs(backend):
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)

    res = sr.vs_ensemble(obs, fct, backend=backend)
    resw = sr.owvs_ensemble(
        obs,
        fct,
        lambda x: backends[backend].mean(x) * 0.0 + 1.0,
        backend=backend,
    )
    np.testing.assert_allclose(res, resw, atol=1e-5)


def test_twvs_vs_vs(backend):
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)

    res = sr.vs_ensemble(obs, fct, backend=backend)
    resw = sr.twvs_ensemble(obs, fct, lambda x: x, backend=backend)
    np.testing.assert_allclose(res, resw, atol=1e-5)


def test_vrvs_vs_vs(backend):
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)

    res = sr.vs_ensemble(obs, fct, backend=backend)
    resw = sr.vrvs_ensemble(
        obs,
        fct,
        lambda x: backends[backend].mean(x) * 0.0 + 1.0,
        backend=backend,
    )
    np.testing.assert_allclose(res, resw, atol=1e-5)


def test_owvariogram_score_correctness(backend):
    fct = np.array(
        [[0.79546742, 0.4777960, 0.2164079], [0.02461368, 0.7584595, 0.3181810]]
    ).T
    obs = np.array([0.2743836, 0.8146400])

    def w_func(x):
        return (
            backends[backend].all(x > 0.2) + 0.0
        )  # + 0.0 works to convert to float in every backend

    res = sr.owvs_ensemble(obs, fct, w_func, p=0.5, backend=backend)
    np.testing.assert_allclose(res, 0.1929739, rtol=1e-6)

    def w_func(x):
        return backends[backend].all(x < 1.0) + 0.0

    res = sr.owvs_ensemble(obs, fct, w_func, p=1.0, backend=backend)
    np.testing.assert_allclose(res, 0.04856366, rtol=1e-6)


def test_twvariogram_score_correctness(backend):
    fct = np.array(
        [[0.79546742, 0.4777960, 0.2164079], [0.02461368, 0.7584595, 0.3181810]]
    ).T
    obs = np.array([0.2743836, 0.8146400])

    def v_func(x):
        return np.maximum(x, 0.2)

    res = sr.twvs_ensemble(obs, fct, v_func, p=0.5, backend=backend)
    np.testing.assert_allclose(res, 0.07594679, rtol=1e-6)

    def v_func(x):
        return np.minimum(x, 1.0)

    res = sr.twvs_ensemble(obs, fct, v_func, p=1.0, backend=backend)
    np.testing.assert_allclose(res, 0.04856366, rtol=1e-6)


@pytest.mark.parametrize("case", MV_CASES)
@pytest.mark.parametrize("M", [6, 7])
@pytest.mark.parametrize("weighted", [False, True], ids=["uw", "w"])
@pytest.mark.parametrize("score", ["owvs_ensemble", "vrvs_ensemble"])
def test_weighted_variogram_paths_agree(score, weighted, M, case, paths_agree, request):
    mark_known_mv_failures(request, case, weighted)
    obs, fct, kwargs = mv_ensemble_inputs(case, M, weighted)
    paths_agree(getattr(sr, score), obs, fct, mv_w_func, **kwargs)


@pytest.mark.parametrize("weighted", [False, True], ids=["uw", "w"])
@pytest.mark.parametrize("score", ["owvs_ensemble", "vrvs_ensemble"])
def test_weighted_variogram_paths_agree_w_p(score, weighted, paths_agree):
    obs, fct, kwargs = mv_ensemble_inputs("default", 7, weighted)
    w = np.random.default_rng(1).uniform(0.1, 2.0, size=(*obs.shape, obs.shape[-1]))
    paths_agree(getattr(sr, score), obs, fct, mv_w_func, w, p=1.0, **kwargs)
