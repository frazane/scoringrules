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


def _vs_w_func(x):
    return 1.0 / (1.0 + x.mean() ** 2)


@pytest.mark.parametrize("score", ["owvs_ensemble", "vrvs_ensemble"])
def test_weighted_variogram_uniform_ens_w(score, backend):
    """Uniform ens_w reproduces the unweighted score."""
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(N, N_VARS))
    fct = rng.normal(size=(N, 6, N_VARS))
    fn = getattr(sr, score)
    res = np.asarray(fn(obs, fct, _vs_w_func, backend=backend))
    res_w = np.asarray(fn(obs, fct, _vs_w_func, ens_w=np.ones((N, 6)), backend=backend))
    np.testing.assert_allclose(res_w, res, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("score", ["owvs_ensemble", "vrvs_ensemble"])
def test_weighted_variogram_ens_w_duplicated_members(score, backend):
    """Integer ens_w equals the unweighted score of the ensemble with each member
    repeated that many times, also with m_axis=0."""
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(N, N_VARS))
    fct = rng.normal(size=(N, 6, N_VARS))
    counts = np.array([1, 3, 1, 2, 1, 1])
    ens_w = np.broadcast_to(counts, (N, 6)).astype(float)
    fn = getattr(sr, score)
    res_dup = np.asarray(
        fn(obs, np.repeat(fct, counts, axis=1), _vs_w_func, backend=backend)
    )
    res_w = np.asarray(fn(obs, fct, _vs_w_func, ens_w=ens_w, backend=backend))
    np.testing.assert_allclose(res_w, res_dup, rtol=1e-5, atol=1e-6)
    res_w = fn(
        obs,
        np.moveaxis(fct, 1, 0),
        _vs_w_func,
        m_axis=0,
        ens_w=ens_w.T,
        backend=backend,
    )
    np.testing.assert_allclose(np.asarray(res_w), res_dup, rtol=1e-5, atol=1e-6)
