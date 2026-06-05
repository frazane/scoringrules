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
    # loose tolerance: jax runs in float32 here, and the two paths accumulate
    # differently, so this guards against gross errors (a real bug differs by O(1)).
    np.testing.assert_allclose(res, res_w, rtol=1e-4, atol=1e-5)


def _nan_policy_check_mv(score_fn, backend):
    """Shared NaN-policy assertions for owes/vres (no estimator argument)."""
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)
    # non-constant weight function: this exercises the weighted denominator and
    # would mask nothing — a constant w_func hides the owes wbar normalisation.
    w_func = lambda x: backends[backend].sum(x**2)  # noqa: E731

    fct_nan = fct.copy()
    fct_nan[0, [0, 3, 6], 0] = np.nan
    fct_nan[2, [5], 1] = np.nan
    nan_positions = np.isnan(fct_nan).any(axis=(1, 2))

    # 'propagate' (default + explicit)
    for policy in (None, "propagate"):
        kw = {} if policy is None else {"nan_policy": policy}
        res = np.asarray(score_fn(obs, fct_nan, w_func, backend=backend, **kw))
        assert np.all(np.isnan(res[nan_positions]))

    # 'raise'
    with pytest.raises(ValueError):
        score_fn(obs, fct_nan, w_func, nan_policy="raise", backend=backend)

    # 'omit': no NaN in result
    res = np.asarray(score_fn(obs, fct_nan, w_func, nan_policy="omit", backend=backend))
    assert not np.any(np.isnan(res))

    # 'omit': equivalence with the surviving sub-ensemble, per row
    for i in range(fct.shape[0]):
        valid = ~np.isnan(fct_nan[i]).any(axis=-1)
        res_clean = score_fn(obs[i], fct_nan[i][valid], w_func, backend=backend)
        res_omit = score_fn(
            obs[i], fct_nan[i], w_func, nan_policy="omit", backend=backend
        )
        assert np.allclose(res_omit, res_clean)

    # NaN in obs always propagates, even under 'omit'
    obs_nan = obs.copy()
    obs_nan[1, 0] = np.nan
    res = np.asarray(score_fn(obs_nan, fct, w_func, nan_policy="omit", backend=backend))
    assert np.isnan(res[1])


def test_owes_ensemble_nan_policy(backend):
    _nan_policy_check_mv(sr.owes_ensemble, backend)


def test_vres_ensemble_nan_policy(backend):
    _nan_policy_check_mv(sr.vres_ensemble, backend)


def test_twes_ensemble_nan_policy(backend):
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)

    fct_nan = fct.copy()
    fct_nan[0, [0, 3, 6], 0] = np.nan
    nan_positions = np.isnan(fct_nan).any(axis=(1, 2))

    # 'propagate'
    res = np.asarray(sr.twes_ensemble(obs, fct_nan, lambda x: x, backend=backend))
    assert np.all(np.isnan(res[nan_positions]))

    # 'raise'
    with pytest.raises(ValueError):
        sr.twes_ensemble(obs, fct_nan, lambda x: x, nan_policy="raise", backend=backend)

    # 'omit': finite, and equals twes over the surviving members (identity v_func)
    res = np.asarray(
        sr.twes_ensemble(obs, fct_nan, lambda x: x, nan_policy="omit", backend=backend)
    )
    assert not np.any(np.isnan(res))
    for i in range(fct.shape[0]):
        valid = ~np.isnan(fct_nan[i]).any(axis=-1)
        res_clean = sr.twes_ensemble(
            obs[i], fct_nan[i][valid], lambda x: x, backend=backend
        )
        res_omit = sr.twes_ensemble(
            obs[i], fct_nan[i], lambda x: x, nan_policy="omit", backend=backend
        )
        assert np.allclose(res_omit, res_clean)
