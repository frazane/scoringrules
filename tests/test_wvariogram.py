import numpy as np
import pytest


import scoringrules as sr

from ._parity import MV_CASES, mv_ensemble_inputs, mv_w_func
from .conftest import assert_inferred


ENSEMBLE_SIZE = 51
N = 100
N_VARS = 3


def test_owvs_vs_vs(native):
    vs = native(sr.vs_ensemble)
    owvs = native(sr.owvs_ensemble)
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)

    res = vs(obs, fct)
    resw = owvs(
        obs,
        fct,
        lambda x: x.mean() * 0.0 + 1.0,
    )
    np.testing.assert_allclose(res, resw, atol=1e-5)


def test_twvs_vs_vs(native):
    vs = native(sr.vs_ensemble)
    twvs = native(sr.twvs_ensemble)
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)

    res = vs(obs, fct)
    resw = twvs(obs, fct, lambda x: x)
    np.testing.assert_allclose(res, resw, atol=1e-5)


def test_vrvs_vs_vs(native):
    vs = native(sr.vs_ensemble)
    vrvs = native(sr.vrvs_ensemble)
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)

    res = vs(obs, fct)
    resw = vrvs(
        obs,
        fct,
        lambda x: x.mean() * 0.0 + 1.0,
    )
    np.testing.assert_allclose(res, resw, atol=1e-5)


def test_owvariogram_score_correctness(native):
    owvs = native(sr.owvs_ensemble)
    fct = np.array(
        [[0.79546742, 0.4777960, 0.2164079], [0.02461368, 0.7584595, 0.3181810]]
    ).T
    obs = np.array([0.2743836, 0.8146400])

    def w_func(x):
        return (x > 0.2).all()

    res = owvs(obs, fct, w_func, p=0.5)
    np.testing.assert_allclose(res, 0.1929739, rtol=1e-6)

    def w_func(x):
        return (x < 1.0).all()

    res = owvs(obs, fct, w_func, p=1.0)
    np.testing.assert_allclose(res, 0.04856366, rtol=1e-6)


def test_twvariogram_score_correctness(native):
    twvs = native(sr.twvs_ensemble)
    fct = np.array(
        [[0.79546742, 0.4777960, 0.2164079], [0.02461368, 0.7584595, 0.3181810]]
    ).T
    obs = np.array([0.2743836, 0.8146400])

    def v_func(x):
        return x.clip(min=0.2)

    res = twvs(obs, fct, v_func, p=0.5)
    np.testing.assert_allclose(res, 0.07594679, rtol=1e-6)

    def v_func(x):
        return x.clip(max=1.0)

    res = twvs(obs, fct, v_func, p=1.0)
    np.testing.assert_allclose(res, 0.04856366, rtol=1e-6)


# no nan_policy: "nan_propagate" checks that NaN propagates
@pytest.mark.parametrize("case", [*MV_CASES, "nan_propagate"])
@pytest.mark.parametrize("M", [6, 7])
@pytest.mark.parametrize("weighted", [False, True], ids=["uw", "w"])
@pytest.mark.parametrize("score", ["owvs_ensemble", "vrvs_ensemble"])
def test_weighted_variogram_paths_agree(score, weighted, M, case, paths_agree):
    obs, fct, kwargs = mv_ensemble_inputs(case, M, weighted)
    kwargs.pop("nan_policy", None)
    paths_agree(getattr(sr, score), obs, fct, mv_w_func, **kwargs)


# p = 1 and 2 take the abs / square branch of the gufuncs, 1.5 the pow branch
@pytest.mark.parametrize("p", [1.0, 2.0, 1.5])
@pytest.mark.parametrize("nan", [False, True], ids=["finite", "nan"])
@pytest.mark.parametrize("weighted", [False, True], ids=["uw", "w"])
@pytest.mark.parametrize("score", ["owvs_ensemble", "vrvs_ensemble"])
def test_weighted_variogram_paths_agree_w_p(nan, p, score, weighted, paths_agree):
    obs, fct, kwargs = mv_ensemble_inputs("default", 7, weighted)
    if nan:  # a NaN member element in sample 1, a NaN obs element in sample 3
        fct[1, 2, 0] = np.nan
        obs[3, 1] = np.nan
    w = np.random.default_rng(1).uniform(0.1, 2.0, size=(*obs.shape, obs.shape[-1]))
    paths_agree(getattr(sr, score), obs, fct, mv_w_func, w, p=p, **kwargs)


def _vs_w_func(x):
    return 1.0 / (1.0 + x.mean() ** 2)


@pytest.mark.parametrize("score", ["owvs_ensemble", "vrvs_ensemble"])
def test_weighted_variogram_uniform_ens_w(score, native):
    """Uniform ens_w reproduces the unweighted score."""
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(N, N_VARS))
    fct = rng.normal(size=(N, 6, N_VARS))
    fn = native(getattr(sr, score))
    res = np.asarray(fn(obs, fct, _vs_w_func))
    res_w = np.asarray(fn(obs, fct, _vs_w_func, ens_w=np.ones((N, 6))))
    np.testing.assert_allclose(res_w, res, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("score", ["owvs_ensemble", "vrvs_ensemble"])
def test_weighted_variogram_ens_w_duplicated_members(score, native):
    """Integer ens_w equals the unweighted score of the ensemble with each member
    repeated that many times, also with m_axis=0."""
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(N, N_VARS))
    fct = rng.normal(size=(N, 6, N_VARS))
    counts = np.array([1, 3, 1, 2, 1, 1])
    ens_w = np.broadcast_to(counts, (N, 6)).astype(float)
    fn = native(getattr(sr, score))
    res_dup = np.asarray(fn(obs, np.repeat(fct, counts, axis=1), _vs_w_func))
    res_w = np.asarray(fn(obs, fct, _vs_w_func, ens_w=ens_w))
    np.testing.assert_allclose(res_w, res_dup, rtol=1e-5, atol=1e-6)
    res_w = fn(
        obs,
        np.moveaxis(fct, 1, 0),
        _vs_w_func,
        m_axis=0,
        ens_w=ens_w.T,
    )
    np.testing.assert_allclose(np.asarray(res_w), res_dup, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("score", ["owvs_ensemble", "vrvs_ensemble"])
def test_weighted_variogram_ens_w_v_axis_first(score, native):
    """ens_w follows fct without its variables axis when v_axis comes first."""
    obs, fct, kwargs = mv_ensemble_inputs("default", 7, weighted=True)
    fn = native(getattr(sr, score))
    expected = fn(obs, fct, _vs_w_func, **kwargs)
    res = fn(
        obs,
        np.swapaxes(fct, -1, -2).copy(),
        _vs_w_func,
        m_axis=-1,
        v_axis=-2,
        **kwargs,
    )
    np.testing.assert_allclose(np.asarray(res), np.asarray(expected), rtol=1e-6)


def test_weighted_variogram_float32(backend, to_backend_float32):
    rng = np.random.default_rng(3)
    obs = to_backend_float32(rng.normal(size=(6, 4)))
    fct = to_backend_float32(rng.normal(size=(6, 7, 4)))
    ens_w = to_backend_float32(rng.uniform(0.1, 1.0, (6, 7)))

    def w_func(x):
        # a boolean weight: the helper casts it to the forecast precision
        return (x > -1.0).all()

    for score in (sr.owvs_ensemble, sr.vrvs_ensemble):
        for kwargs in ({}, {"ens_w": ens_w}):
            res = score(obs, fct, w_func, **kwargs)
            assert_inferred(res, backend)
            assert res.dtype == fct.dtype


@pytest.mark.parametrize("score", ["owvs_ensemble", "vrvs_ensemble"])
def test_weighted_variogram_grad_torch(score):
    """p=1, see test_variogram_grad_torch."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(4)
    obs = torch.as_tensor(rng.normal(size=(4, 3)))
    fct = torch.tensor(rng.normal(size=(4, 7, 3)), requires_grad=True)
    ens_w = torch.as_tensor(rng.uniform(0.1, 1.0, (4, 7)))
    fn = getattr(sr, score)
    for kwargs in ({}, {"ens_w": ens_w}):
        fct.grad = None
        fn(
            obs, fct, lambda x: torch.sigmoid(x.mean()), p=1.0, **kwargs
        ).sum().backward()
        assert fct.grad is not None and torch.isfinite(fct.grad).all()
