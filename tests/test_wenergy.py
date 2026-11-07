import numpy as np
import pytest

import scoringrules as sr

from ._parity import MV_CASES, mv_ensemble_inputs, mv_w_func
from .conftest import assert_inferred


ENSEMBLE_SIZE = 51
N = 100
N_VARS = 3


def test_owes_vs_es(backend_kwargs, native):
    es = native(sr.es_ensemble)
    owes = native(sr.owes_ensemble)
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)

    res = es(obs, fct, **backend_kwargs)
    resw = owes(
        obs,
        fct,
        lambda x: x.mean() * 0.0 + 1.0,
        **backend_kwargs,
    )
    np.testing.assert_allclose(res, resw, atol=1e-6)


def test_twes_vs_es(backend_kwargs, native):
    es = native(sr.es_ensemble)
    twes = native(sr.twes_ensemble)
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)

    res = es(obs, fct, **backend_kwargs)
    resw = twes(obs, fct, lambda x: x, **backend_kwargs)
    np.testing.assert_allclose(res, resw, rtol=1e-10)


def test_vres_vs_es(backend_kwargs, native):
    es = native(sr.es_ensemble)
    vres = native(sr.vres_ensemble)
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)

    res = es(obs, fct, **backend_kwargs)
    resw = vres(
        obs,
        fct,
        lambda x: x.mean() * 0.0 + 1.0,
        **backend_kwargs,
    )
    np.testing.assert_allclose(res, resw, rtol=1e-6)


def test_owenergy_score_correctness(backend_kwargs, native):
    owes = native(sr.owes_ensemble)
    fct = np.array(
        [[0.79546742, 0.4777960, 0.2164079], [0.02461368, 0.7584595, 0.3181810]]
    ).T
    obs = np.array([0.2743836, 0.8146400])

    def w_func(x):
        return (x > 0.2).all()

    res = owes(obs, fct, w_func, **backend_kwargs)
    np.testing.assert_allclose(res, 0.2274243, rtol=1e-6)

    def w_func(x):
        return (x < 1.0).all()

    res = owes(obs, fct, w_func, **backend_kwargs)
    np.testing.assert_allclose(res, 0.3345418, rtol=1e-6)


def test_twenergy_score_correctness(backend_kwargs, native):
    twes = native(sr.twes_ensemble)
    fct = np.array(
        [[0.79546742, 0.4777960, 0.2164079], [0.02461368, 0.7584595, 0.3181810]]
    ).T
    obs = np.array([0.2743836, 0.8146400])

    def v_func(x):
        return x.clip(min=0.2)

    res = twes(obs, fct, v_func, **backend_kwargs)
    np.testing.assert_allclose(res, 0.3116075, rtol=1e-6)

    def v_func(x):
        return x.clip(max=1.0)

    res = twes(obs, fct, v_func, **backend_kwargs)
    np.testing.assert_allclose(res, 0.3345418, rtol=1e-6)


@pytest.mark.parametrize("score_fn", [sr.owes_ensemble, sr.vres_ensemble])
def test_weighted_energy_uniform_ens_w_equivalence(score_fn, backend_kwargs, native):
    """Uniform ens_w must reproduce the unweighted score. This guards the
    weighted numba gufuncs (_owenergy_score_gufunc_w / _vrenergy_score_gufunc_w)
    against the trusted unweighted kernels."""
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)
    w_func = lambda x: x.mean() * 0.0 + 1.0  # noqa: E731
    ens_w = np.ones(fct.shape[:-1])
    score_fn = native(score_fn)
    res = np.asarray(score_fn(obs, fct, w_func, **backend_kwargs))
    res_w = np.asarray(score_fn(obs, fct, w_func, ens_w=ens_w, **backend_kwargs))
    # loose tolerance: jax runs in float32 here, and the two paths accumulate
    # differently, so this guards against gross errors (a real bug differs by O(1)).
    np.testing.assert_allclose(res, res_w, rtol=1e-4, atol=1e-5)


def _nan_policy_check_mv(score_fn, backend_kwargs):
    """Shared NaN-policy assertions for owes/vres (no estimator argument).
    ``score_fn`` is wrapped by the ``native`` fixture."""
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)
    # non-constant weight function: this exercises the weighted denominator and
    # would mask nothing — a constant w_func hides the owes wbar normalisation.
    w_func = lambda x: (x**2).sum()  # noqa: E731

    fct_nan = fct.copy()
    fct_nan[0, [0, 3, 6], 0] = np.nan
    fct_nan[2, [5], 1] = np.nan
    nan_positions = np.isnan(fct_nan).any(axis=(1, 2))

    # 'propagate' (default + explicit)
    for policy in (None, "propagate"):
        kw = {} if policy is None else {"nan_policy": policy}
        res = np.asarray(score_fn(obs, fct_nan, w_func, **kw, **backend_kwargs))
        assert np.all(np.isnan(res[nan_positions]))

    # 'raise'
    with pytest.raises(ValueError):
        score_fn(obs, fct_nan, w_func, nan_policy="raise", **backend_kwargs)

    # 'omit': no NaN in result
    res = np.asarray(
        score_fn(obs, fct_nan, w_func, nan_policy="omit", **backend_kwargs)
    )
    assert not np.any(np.isnan(res))

    # 'omit': equivalence with the surviving sub-ensemble, per row
    for i in range(fct.shape[0]):
        valid = ~np.isnan(fct_nan[i]).any(axis=-1)
        res_clean = score_fn(obs[i], fct_nan[i][valid], w_func, **backend_kwargs)
        res_omit = score_fn(
            obs[i], fct_nan[i], w_func, nan_policy="omit", **backend_kwargs
        )
        assert np.allclose(res_omit, res_clean)

    # NaN in obs always propagates, even under 'omit'
    obs_nan = obs.copy()
    obs_nan[1, 0] = np.nan
    res = np.asarray(
        score_fn(obs_nan, fct, w_func, nan_policy="omit", **backend_kwargs)
    )
    assert np.isnan(res[1])


def test_owes_ensemble_nan_policy(backend_kwargs, native):
    _nan_policy_check_mv(native(sr.owes_ensemble), backend_kwargs)


def test_vres_ensemble_nan_policy(backend_kwargs, native):
    _nan_policy_check_mv(native(sr.vres_ensemble), backend_kwargs)


def test_twes_ensemble_nan_policy(backend_kwargs, native):
    twes = native(sr.twes_ensemble)
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)

    fct_nan = fct.copy()
    fct_nan[0, [0, 3, 6], 0] = np.nan
    nan_positions = np.isnan(fct_nan).any(axis=(1, 2))

    # 'propagate'
    res = np.asarray(twes(obs, fct_nan, lambda x: x, **backend_kwargs))
    assert np.all(np.isnan(res[nan_positions]))

    # 'raise'
    with pytest.raises(ValueError):
        twes(obs, fct_nan, lambda x: x, nan_policy="raise", **backend_kwargs)

    # 'omit': finite, and equals twes over the surviving members (identity v_func)
    res = np.asarray(
        twes(obs, fct_nan, lambda x: x, nan_policy="omit", **backend_kwargs)
    )
    assert not np.any(np.isnan(res))
    for i in range(fct.shape[0]):
        valid = ~np.isnan(fct_nan[i]).any(axis=-1)
        res_clean = twes(obs[i], fct_nan[i][valid], lambda x: x, **backend_kwargs)
        res_omit = twes(
            obs[i], fct_nan[i], lambda x: x, nan_policy="omit", **backend_kwargs
        )
        assert np.allclose(res_omit, res_clean)

    # NaN in obs always propagates, even under 'omit'
    obs_nan = obs.copy()
    obs_nan[1, 0] = np.nan
    res = np.asarray(
        twes(obs_nan, fct, lambda x: x, nan_policy="omit", **backend_kwargs)
    )
    assert np.isnan(res[1])


@pytest.mark.parametrize("score_fn", [sr.owes_ensemble, sr.vres_ensemble])
def test_weighted_energy_nan_policy_with_ens_w(score_fn, backend_kwargs, native):
    """owes/vres under 'omit' WITH user-supplied ens_w. Exercises the weighted
    estimators (incl. the fixed numba kernels) under the exact scenario that
    motivated the wbar / ow-indexing bugfixes."""
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)
    score_fn = native(score_fn)
    w_func = lambda x: (x**2).sum()  # noqa: E731
    ens_w = np.ones(fct.shape[:-1])

    fct_nan = fct.copy()
    fct_nan[0, [0, 3, 6], 0] = np.nan
    fct_nan[2, [5], 1] = np.nan

    # 'omit': no NaN in result
    res = np.asarray(
        score_fn(obs, fct_nan, w_func, ens_w=ens_w, nan_policy="omit", **backend_kwargs)
    )
    assert not np.any(np.isnan(res))

    # 'omit': equivalence with the clean weighted sub-ensemble, per row
    for i in range(fct.shape[0]):
        valid = ~np.isnan(fct_nan[i]).any(axis=-1)
        res_clean = score_fn(
            obs[i],
            fct_nan[i][valid],
            w_func,
            ens_w=ens_w[i][valid],
            nan_policy="omit",
            **backend_kwargs,
        )
        res_omit = score_fn(
            obs[i],
            fct_nan[i],
            w_func,
            ens_w=ens_w[i],
            nan_policy="omit",
            **backend_kwargs,
        )
        assert np.allclose(res_omit, res_clean)


@pytest.mark.parametrize("case", MV_CASES)
@pytest.mark.parametrize("M", [6, 7])
@pytest.mark.parametrize("weighted", [False, True], ids=["uw", "w"])
@pytest.mark.parametrize("score", ["owes_ensemble", "vres_ensemble"])
def test_weighted_energy_paths_agree(score, weighted, M, case, paths_agree):
    obs, fct, kwargs = mv_ensemble_inputs(case, M, weighted)
    paths_agree(getattr(sr, score), obs, fct, mv_w_func, **kwargs)


@pytest.mark.parametrize("score", ["owes_ensemble", "vres_ensemble"])
def test_weighted_energy_ens_w_v_axis_first(score, native, backend_kwargs):
    """ens_w follows fct without its variables axis, also when v_axis comes
    before m_axis, with and without the nan-policy step."""
    obs, fct, kwargs = mv_ensemble_inputs("default", 7, weighted=True)
    fn = native(getattr(sr, score))
    w_func = lambda x: 1.0 / (1.0 + (x**2).mean())  # noqa: E731
    expected = np.asarray(fn(obs, fct, w_func, **kwargs, **backend_kwargs))
    fct_t = np.swapaxes(fct, -1, -2).copy()
    for nan_policy in ("propagate", "omit"):
        res = fn(
            obs,
            fct_t,
            w_func,
            m_axis=-1,
            v_axis=-2,
            nan_policy=nan_policy,
            **kwargs,
            **backend_kwargs,
        )
        np.testing.assert_allclose(np.asarray(res), expected, rtol=1e-6)


def test_weighted_energy_float32(backend, to_backend_float32, backend_kwargs):
    rng = np.random.default_rng(3)
    obs = to_backend_float32(rng.normal(size=(N, N_VARS)))
    fct = to_backend_float32(rng.normal(size=(N, ENSEMBLE_SIZE, N_VARS)))
    ens_w = to_backend_float32(rng.uniform(0.1, 1.0, (N, ENSEMBLE_SIZE)))

    def w_func(x):
        # a boolean weight: the helper casts it to the forecast precision
        return (x > -1.0).all()

    for score in (sr.owes_ensemble, sr.vres_ensemble):
        for kwargs in ({}, {"ens_w": ens_w}):
            res = score(obs, fct, w_func, **kwargs, **backend_kwargs)
            assert_inferred(res, backend)
            assert res.dtype == fct.dtype


@pytest.mark.parametrize("score", ["owes_ensemble", "vres_ensemble"])
def test_weighted_energy_grad_torch(score):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(4)
    obs = torch.as_tensor(rng.normal(size=(4, N_VARS)))
    fct = torch.tensor(rng.normal(size=(4, 9, N_VARS)), requires_grad=True)
    ens_w = torch.as_tensor(rng.uniform(0.1, 1.0, (4, 9)))
    fn = getattr(sr, score)
    for kwargs in ({}, {"ens_w": ens_w}):
        fct.grad = None
        fn(obs, fct, lambda x: torch.sigmoid(x.mean()), **kwargs).sum().backward()
        assert fct.grad is not None and torch.isfinite(fct.grad).all()
    fct.grad = None
    sr.twes_ensemble(obs, fct, lambda x: x.clip(min=-0.5)).sum().backward()
    assert fct.grad is not None and torch.isfinite(fct.grad).all()


@pytest.mark.parametrize(
    "score", ["owes_ensemble", "vres_ensemble", "owvs_ensemble", "vrvs_ensemble"]
)
def test_weighted_mv_python_float_w_func_torch(score):
    """A w_func returning Python floats (not vmappable: ``.item()``) runs through
    the torch loop fallback and keeps float64 precision (matches numpy)."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(6)
    obs = rng.normal(size=(4, N_VARS))
    fct = rng.normal(size=(4, 9, N_VARS))

    def w_func(x):
        return float(1.0 / (1.0 + np.exp(-float(x.mean().item()))))

    fn = getattr(sr, score)
    expected = fn(obs, fct, w_func)
    res = fn(torch.as_tensor(obs), torch.as_tensor(fct), w_func)
    assert res.dtype == torch.float64
    np.testing.assert_allclose(res.numpy(), expected, rtol=1e-13)
