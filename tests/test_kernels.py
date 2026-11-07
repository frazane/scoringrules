import math

import numpy as np
import pytest
import scoringrules as sr
from scoringrules.backend import backends

from ._parity import (
    MV_CASES,
    UV_CASES,
    mark_known_mv_failures,
    mv_ensemble_inputs,
    mv_v_func,
    mv_w_func,
    uv_ensemble_inputs,
    uv_w_func,
)


ENSEMBLE_SIZE = 11
N = 10
N_VARS = 3

ESTIMATORS = ["nrg", "fair", "akr", "akr_circperm"]


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_gksuv(estimator, backend):
    obs = np.random.randn(N)
    mu = obs + np.random.randn(N) * 0.1
    sigma = abs(np.random.randn(N)) * 0.3
    fct = np.random.randn(N, ENSEMBLE_SIZE) * sigma[..., None] + mu[..., None]

    # undefined estimator
    with pytest.raises(ValueError):
        est = "undefined_estimator"
        sr.gksuv_ensemble(obs, fct, estimator=est, backend=backend)

    # m_axis keyword
    res = sr.gksuv_ensemble(obs, fct, estimator=estimator, backend=backend)
    res2 = sr.gksuv_ensemble(obs, fct.T, m_axis=0, estimator=estimator, backend=backend)
    assert np.allclose(res, res2)

    if estimator == "nrg":
        # non-negative values
        res = sr.gksuv_ensemble(obs, fct, estimator=estimator, backend=backend)
        res = np.asarray(res)
        assert not np.any(res < 0.0)

        # m_axis keyword
        res = sr.gksuv_ensemble(
            obs,
            np.random.randn(ENSEMBLE_SIZE, N),
            m_axis=0,
            estimator=estimator,
            backend=backend,
        )
        res = np.asarray(res)
        assert not np.any(res < 0.0)

        # approx zero when perfect forecast
        perfect_fct = obs[..., None] + np.random.randn(N, ENSEMBLE_SIZE) * 0.00001
        res = sr.gksuv_ensemble(obs, perfect_fct, estimator=estimator, backend=backend)
        res = np.asarray(res)
        assert not np.any(res - 0.0 > 0.0001)

    # test correctness
    obs, fct = 11.6, np.array([9.8, 8.7, 11.9, 12.1, 13.4])

    if estimator == "nrg":
        res = sr.gksuv_ensemble(obs, fct, estimator=estimator, backend=backend)
        expected = 0.2490516
        assert np.isclose(res, expected)

    elif estimator == "fair":
        res = sr.gksuv_ensemble(obs, fct, estimator=estimator, backend=backend)
        expected = 0.1737752
        assert np.isclose(res, expected)

    elif estimator == "akr":
        res = sr.gksuv_ensemble(obs, fct, estimator=estimator, backend=backend)
        expected = 0.2464915
        assert np.isclose(res, expected)

    elif estimator == "akr_circperm":
        res = sr.gksuv_ensemble(obs, fct, estimator=estimator, backend=backend)
        expected = 0.1010588
        assert np.isclose(res, expected)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_gksmv(estimator, backend):
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)

    # undefined estimator
    with pytest.raises(ValueError):
        est = "undefined_estimator"
        sr.gksmv_ensemble(obs, fct, estimator=est, backend=backend)

    res = sr.gksmv_ensemble(obs, fct, estimator=estimator, backend=backend)

    if backend in ["numpy", "numba"]:
        assert isinstance(res, np.ndarray)
    elif backend == "jax":
        assert "jax" in res.__module__

    # approx zero when perfect forecast
    if estimator == "nrg":
        perfect_fct = (
            np.expand_dims(obs, axis=-2)
            + np.random.randn(N, ENSEMBLE_SIZE, N_VARS) * 0.00001
        )
        res = sr.gksmv_ensemble(obs, perfect_fct, estimator=estimator, backend=backend)
        res = np.asarray(res)
        assert not np.any(res - 0.0 > 0.0001)

    # test correctness
    obs = np.array([11.6, -23.1])
    fct = np.array(
        [[9.8, 8.7, 11.9, 12.1, 13.4], [-24.8, -18.5, -29.9, -18.3, -21.0]]
    ).transpose()

    # test exceptions
    with pytest.raises(ValueError):
        est = "undefined_estimator"
        sr.gksmv_ensemble(obs, fct, estimator=est, backend=backend)

    with pytest.raises(ValueError):
        est = "undefined_estimator"
        sr.gksmv_ensemble(obs, fct, estimator=est, backend=backend)

    if estimator == "nrg":
        res = sr.gksmv_ensemble(obs, fct, estimator=estimator, backend=backend)
        expected = 0.5868737
        assert np.isclose(res, expected)

    elif estimator == "fair":
        res = sr.gksmv_ensemble(obs, fct, estimator=estimator, backend=backend)
        expected = 0.4870162
        assert np.isclose(res, expected)

    elif estimator == "akr":
        res = sr.gksmv_ensemble(obs, fct, estimator=estimator, backend=backend)
        expected = 0.4874259
        assert np.isclose(res, expected)

    elif estimator == "akr_circperm":
        res = sr.gksmv_ensemble(obs, fct, estimator=estimator, backend=backend)
        expected = 0.4866066
        assert np.isclose(res, expected)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_twgksuv(estimator, backend):
    if backend == "jax":
        pytest.skip("Not implemented in jax backend")
    obs = np.random.randn(N)
    mu = obs + np.random.randn(N) * 0.1
    sigma = abs(np.random.randn(N)) * 0.3
    fct = np.random.randn(N, ENSEMBLE_SIZE) * sigma[..., None] + mu[..., None]

    res = sr.gksuv_ensemble(obs, fct, backend=backend, estimator=estimator)

    # no argument given
    resw = sr.twgksuv_ensemble(obs, fct, estimator=estimator, backend=backend)
    np.testing.assert_allclose(res, resw, rtol=1e-10)

    # a and b
    resw = sr.twgksuv_ensemble(
        obs, fct, a=float("-inf"), b=float("inf"), estimator=estimator, backend=backend
    )
    np.testing.assert_allclose(res, resw, rtol=1e-10)

    # v_func as identity function
    resw = sr.twgksuv_ensemble(
        obs, fct, v_func=lambda x: x, estimator=estimator, backend=backend
    )
    np.testing.assert_allclose(res, resw, rtol=1e-10)

    # test correctness
    fct = np.array(
        [
            [-0.03574194, 0.06873582, 0.03098684, 0.07316138, 0.08498165],
            [-0.11957874, 0.26472238, -0.06324622, 0.43026451, -0.25640457],
            [-1.31340831, -1.43722153, -1.01696021, -0.70790148, -1.20432392],
            [1.26054027, 1.03477874, 0.61688454, 0.75305795, 1.19364529],
            [0.63192933, 0.33521695, 0.20626017, 0.43158264, 0.69518928],
            [0.83305233, 0.71965134, 0.96687378, 1.0000473, 0.82714425],
            [0.0128655, 0.03849841, 0.02459106, -0.02618909, 0.18687008],
            [-0.69399216, -0.59432627, -1.43629972, 0.01979857, -0.3286767],
            [1.92958764, 2.2678255, 1.86036922, 1.84766384, 2.03331138],
            [0.80847492, 1.11265193, 0.58995365, 1.04838184, 1.10439277],
        ]
    )

    obs = np.array(
        [
            0.19640722,
            -0.11300369,
            -1.0400268,
            0.84306533,
            0.36572078,
            0.82487264,
            0.14088773,
            -0.51936113,
            1.70706264,
            0.75985908,
        ]
    )

    def v_func1(x):
        return np.maximum(x, -1.0)

    def v_func2(x):
        return np.minimum(x, 1.85)

    if estimator == "nrg":
        res = np.mean(
            np.float64(
                sr.twgksuv_ensemble(
                    obs, fct, v_func=v_func1, estimator=estimator, backend=backend
                )
            )
        )
        np.testing.assert_allclose(res, 0.01018774, rtol=1e-6)

        res = np.mean(
            np.float64(
                sr.twgksuv_ensemble(
                    obs, fct, a=-1.0, estimator=estimator, backend=backend
                )
            )
        )
        np.testing.assert_allclose(res, 0.01018774, rtol=1e-6)

        res = np.mean(
            np.float64(
                sr.twgksuv_ensemble(
                    obs, fct, v_func=v_func2, estimator=estimator, backend=backend
                )
            )
        )
        np.testing.assert_allclose(res, 0.0089314, rtol=1e-6)

        res = np.mean(
            np.float64(
                sr.twgksuv_ensemble(
                    obs, fct, b=1.85, estimator=estimator, backend=backend
                )
            )
        )
        np.testing.assert_allclose(res, 0.0089314, rtol=1e-6)


def test_twgksmv(backend):
    if backend == "jax":
        pytest.skip("Not implemented in jax backend")
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)

    res0 = sr.gksmv_ensemble(obs, fct, backend=backend)
    res = sr.twgksmv_ensemble(obs, fct, lambda x: x, backend=backend)
    np.testing.assert_allclose(res, res0, rtol=1e-6)

    fct = np.array(
        [[0.79546742, 0.4777960, 0.2164079], [0.02461368, 0.7584595, 0.3181810]]
    ).T
    obs = np.array([0.2743836, 0.8146400])

    def v_func(x):
        return np.maximum(x, 0.2)

    res = sr.twgksmv_ensemble(obs, fct, v_func, backend=backend)
    np.testing.assert_allclose(res, 0.08671498, rtol=1e-6)

    def v_func(x):
        return np.minimum(x, 1)

    res = sr.twgksmv_ensemble(obs, fct, v_func, backend=backend)
    np.testing.assert_allclose(res, 0.1016436, rtol=1e-6)


def test_owgksuv(backend):
    if backend == "jax":
        pytest.skip("Not implemented in jax backend")
    obs = np.random.randn(N)
    mu = obs + np.random.randn(N) * 0.1
    sigma = abs(np.random.randn(N)) * 0.3
    fct = np.random.randn(N, ENSEMBLE_SIZE) * sigma[..., None] + mu[..., None]

    # m_axis keyword
    res = sr.owgksuv_ensemble(
        obs,
        np.random.randn(ENSEMBLE_SIZE, N),
        w_func=lambda x: x * 0.0 + 1.0,
        m_axis=0,
        backend=backend,
    )
    res = np.asarray(res)
    assert not np.any(res < 0.0)

    res = sr.gksuv_ensemble(obs, fct, backend=backend, estimator="nrg")

    # no argument given
    resw = sr.owgksuv_ensemble(obs, fct, backend=backend)
    np.testing.assert_allclose(res, resw, atol=1e-6)

    # a and b
    resw = sr.owgksuv_ensemble(
        obs, fct, a=float("-inf"), b=float("inf"), backend=backend
    )
    np.testing.assert_allclose(res, resw, atol=1e-6)

    # w_func as identity function
    resw = sr.owgksuv_ensemble(
        obs, fct, w_func=lambda x: x * 0.0 + 1.0, backend=backend
    )
    np.testing.assert_allclose(res, resw, atol=1e-6)

    # test correctness
    fct = np.array(
        [
            [-0.03574194, 0.06873582, 0.03098684, 0.07316138, 0.08498165],
            [-0.11957874, 0.26472238, -0.06324622, 0.43026451, -0.25640457],
            [-1.31340831, -1.43722153, -1.01696021, -0.70790148, -1.20432392],
            [1.26054027, 1.03477874, 0.61688454, 0.75305795, 1.19364529],
            [0.63192933, 0.33521695, 0.20626017, 0.43158264, 0.69518928],
            [0.83305233, 0.71965134, 0.96687378, 1.0000473, 0.82714425],
            [0.0128655, 0.03849841, 0.02459106, -0.02618909, 0.18687008],
            [-0.69399216, -0.59432627, -1.43629972, 0.01979857, -0.3286767],
            [1.92958764, 2.2678255, 1.86036922, 1.84766384, 2.03331138],
            [0.80847492, 1.11265193, 0.58995365, 1.04838184, 1.10439277],
        ]
    )

    obs = np.array(
        [
            0.19640722,
            -0.11300369,
            -1.0400268,
            0.84306533,
            0.36572078,
            0.82487264,
            0.14088773,
            -0.51936113,
            1.70706264,
            0.75985908,
        ]
    )

    def w_func(x):
        return (x > -1) * 1.0

    res = np.mean(
        np.float64(sr.owgksuv_ensemble(obs, fct, w_func=w_func, backend=backend))
    )
    np.testing.assert_allclose(res, 0.01036335, rtol=1e-5)

    res = np.mean(np.float64(sr.owgksuv_ensemble(obs, fct, a=-1.0, backend=backend)))
    np.testing.assert_allclose(res, 0.01036335, rtol=1e-5)

    def w_func(x):
        return (x < 1.85) * 1.0

    res = np.mean(
        np.float64(sr.owgksuv_ensemble(obs, fct, w_func=w_func, backend=backend))
    )
    np.testing.assert_allclose(res, 0.008905213, rtol=1e-5)

    res = np.mean(np.float64(sr.owgksuv_ensemble(obs, fct, b=1.85, backend=backend)))
    np.testing.assert_allclose(res, 0.008905213, rtol=1e-5)


def test_owgksmv(backend):
    if backend == "jax":
        pytest.skip("Not implemented in jax backend")
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)

    res0 = sr.gksmv_ensemble(obs, fct, backend=backend)
    res = sr.owgksmv_ensemble(
        obs, fct, lambda x: backends[backend].mean(x) * 0.0 + 1.0, backend=backend
    )
    np.testing.assert_allclose(res, res0, rtol=1e-6)

    fct = np.array(
        [[0.79546742, 0.4777960, 0.2164079], [0.02461368, 0.7584595, 0.3181810]]
    ).T
    obs = np.array([0.2743836, 0.8146400])

    def w_func(x):
        return backends[backend].all(x > 0.2)

    res = sr.owgksmv_ensemble(obs, fct, w_func, backend=backend)
    np.testing.assert_allclose(res, 0.03901075, rtol=1e-5)

    def w_func(x):
        return backends[backend].all(x < 1.0)

    res = sr.owgksmv_ensemble(obs, fct, w_func, backend=backend)
    np.testing.assert_allclose(res, 0.1016436, rtol=1e-6)


def test_vrgksuv(backend):
    if backend == "jax":
        pytest.skip("Not implemented in jax backend")
    obs = np.random.randn(N)
    mu = obs + np.random.randn(N) * 0.1
    sigma = abs(np.random.randn(N)) * 0.3
    fct = np.random.randn(N, ENSEMBLE_SIZE) * sigma[..., None] + mu[..., None]

    # m_axis keyword
    res = sr.vrgksuv_ensemble(
        obs,
        np.random.randn(ENSEMBLE_SIZE, N),
        w_func=lambda x: x * 0.0 + 1.0,
        m_axis=0,
        backend=backend,
    )
    res = np.asarray(res)
    assert not np.any(res < 0.0)

    res = sr.gksuv_ensemble(obs, fct, backend=backend, estimator="nrg")

    # no argument given
    resw = sr.vrgksuv_ensemble(obs, fct, backend=backend)
    np.testing.assert_allclose(res, resw, rtol=1e-5)

    # a and b
    resw = sr.vrgksuv_ensemble(
        obs, fct, a=float("-inf"), b=float("inf"), backend=backend
    )
    np.testing.assert_allclose(res, resw, rtol=1e-5)

    # w_func as identity function
    resw = sr.vrgksuv_ensemble(
        obs, fct, w_func=lambda x: x * 0.0 + 1.0, backend=backend
    )
    np.testing.assert_allclose(res, resw, rtol=1e-5)

    # test correctness
    fct = np.array(
        [
            [-0.03574194, 0.06873582, 0.03098684, 0.07316138, 0.08498165],
            [-0.11957874, 0.26472238, -0.06324622, 0.43026451, -0.25640457],
            [-1.31340831, -1.43722153, -1.01696021, -0.70790148, -1.20432392],
            [1.26054027, 1.03477874, 0.61688454, 0.75305795, 1.19364529],
            [0.63192933, 0.33521695, 0.20626017, 0.43158264, 0.69518928],
            [0.83305233, 0.71965134, 0.96687378, 1.0000473, 0.82714425],
            [0.0128655, 0.03849841, 0.02459106, -0.02618909, 0.18687008],
            [-0.69399216, -0.59432627, -1.43629972, 0.01979857, -0.3286767],
            [1.92958764, 2.2678255, 1.86036922, 1.84766384, 2.03331138],
            [0.80847492, 1.11265193, 0.58995365, 1.04838184, 1.10439277],
        ]
    )

    obs = np.array(
        [
            0.19640722,
            -0.11300369,
            -1.0400268,
            0.84306533,
            0.36572078,
            0.82487264,
            0.14088773,
            -0.51936113,
            1.70706264,
            0.75985908,
        ]
    )

    def w_func(x):
        return (x > -1) * 1.0

    res = np.mean(
        np.float64(sr.vrgksuv_ensemble(obs, fct, w_func=w_func, backend=backend))
    )
    np.testing.assert_allclose(res, 0.01476682, rtol=1e-6)

    res = np.mean(np.float64(sr.vrgksuv_ensemble(obs, fct, a=-1.0, backend=backend)))
    np.testing.assert_allclose(res, 0.01476682, rtol=1e-6)

    def w_func(x):
        return (x < 1.85) * 1.0

    res = np.mean(
        np.float64(sr.vrgksuv_ensemble(obs, fct, w_func=w_func, backend=backend))
    )
    np.testing.assert_allclose(res, 0.04011836, rtol=1e-6)

    res = np.mean(np.float64(sr.vrgksuv_ensemble(obs, fct, b=1.85, backend=backend)))
    np.testing.assert_allclose(res, 0.04011836, rtol=1e-6)


def test_vrgksmv(backend):
    if backend == "jax":
        pytest.skip("Not implemented in jax backend")
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)

    res0 = sr.gksmv_ensemble(obs, fct, backend=backend)
    res = sr.vrgksmv_ensemble(
        obs, fct, lambda x: backends[backend].mean(x) * 0.0 + 1.0, backend=backend
    )
    np.testing.assert_allclose(res, res0, rtol=1e-6)


# the univariate kernel scores have no nan_policy
_KS_UV_CASES = [c for c in UV_CASES if not c.startswith("nan_")]


@pytest.mark.parametrize("case", _KS_UV_CASES)
@pytest.mark.parametrize("M", [6, 7])
@pytest.mark.parametrize("weighted", [False, True], ids=["uw", "w"])
@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("score", ["gksuv_ensemble", "twgksuv_ensemble"])
def test_gksuv_paths_agree(score, estimator, weighted, M, case, paths_agree):
    obs, fct, kwargs = uv_ensemble_inputs(case, M, weighted)
    args = (-0.5, 1.0) if score == "twgksuv_ensemble" else ()
    paths_agree(getattr(sr, score), obs, fct, *args, estimator=estimator, **kwargs)


@pytest.mark.parametrize("case", _KS_UV_CASES)
@pytest.mark.parametrize("M", [6, 7])
@pytest.mark.parametrize("weighted", [False, True], ids=["uw", "w"])
@pytest.mark.parametrize("score", ["owgksuv_ensemble", "vrgksuv_ensemble"])
def test_weighted_gksuv_paths_agree(score, weighted, M, case, paths_agree):
    obs, fct, kwargs = uv_ensemble_inputs(case, M, weighted)
    paths_agree(getattr(sr, score), obs, fct, w_func=uv_w_func, **kwargs)


@pytest.mark.parametrize("case", MV_CASES)
@pytest.mark.parametrize("M", [6, 7])
@pytest.mark.parametrize("weighted", [False, True], ids=["uw", "w"])
@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("score", ["gksmv_ensemble", "twgksmv_ensemble"])
def test_gksmv_paths_agree(score, estimator, weighted, M, case, paths_agree, request):
    mark_known_mv_failures(request, case, weighted)
    obs, fct, kwargs = mv_ensemble_inputs(case, M, weighted)
    args = (mv_v_func,) if score == "twgksmv_ensemble" else ()
    paths_agree(getattr(sr, score), obs, fct, *args, estimator=estimator, **kwargs)


@pytest.mark.parametrize("case", MV_CASES)
@pytest.mark.parametrize("M", [6, 7])
@pytest.mark.parametrize("weighted", [False, True], ids=["uw", "w"])
@pytest.mark.parametrize("score", ["owgksmv_ensemble", "vrgksmv_ensemble"])
def test_weighted_gksmv_paths_agree(score, weighted, M, case, paths_agree, request):
    mark_known_mv_failures(request, case, weighted)
    obs, fct, kwargs = mv_ensemble_inputs(case, M, weighted)
    paths_agree(getattr(sr, score), obs, fct, mv_w_func, **kwargs)


def _uv_w_func(x):
    return 1.0 / (1.0 + x**2)


def _mv_w_func(x):
    return 1.0 / (1.0 + x.mean() ** 2)


def _gks_ens_w_inputs(mv, seed=0):
    """Inputs plus integer member counts and the ensemble with duplicated members."""
    rng = np.random.default_rng(seed)
    M = 6
    obs = rng.normal(size=(N, N_VARS) if mv else (N,))
    fct = rng.normal(size=(N, M, N_VARS) if mv else (N, M))
    counts = np.array([1, 3, 1, 2, 1, 1])
    fct_dup = np.repeat(fct, counts, axis=1)
    ens_w = np.broadcast_to(counts, (N, M)).astype(float)
    return obs, fct, counts, fct_dup, ens_w


_GKS_W_SCORES = [
    *[("gksuv_ensemble", est) for est in ESTIMATORS],
    *[("gksmv_ensemble", est) for est in ESTIMATORS],
    ("owgksuv_ensemble", None),
    ("vrgksuv_ensemble", None),
    ("owgksmv_ensemble", None),
    ("vrgksmv_ensemble", None),
]


def _gks_call(score, estimator, obs, fct, **kwargs):
    fn = getattr(sr, score)
    if estimator is not None:
        return np.asarray(fn(obs, fct, estimator=estimator, **kwargs))
    if score.endswith("uv_ensemble"):
        return np.asarray(fn(obs, fct, w_func=_uv_w_func, **kwargs))
    return np.asarray(fn(obs, fct, _mv_w_func, **kwargs))


@pytest.mark.parametrize("score,estimator", _GKS_W_SCORES)
def test_gks_uniform_ens_w(score, estimator, backend):
    """Uniform ens_w reproduces the unweighted score."""
    obs, fct, _, _, ens_w = _gks_ens_w_inputs(score.endswith("mv_ensemble"))
    ens_w = np.ones_like(ens_w)
    res = _gks_call(score, estimator, obs, fct, backend=backend)
    res_w = _gks_call(score, estimator, obs, fct, ens_w=ens_w, backend=backend)
    np.testing.assert_allclose(res_w, res, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "score,estimator", [s for s in _GKS_W_SCORES if s[1] in (None, "nrg")]
)
def test_gks_ens_w_duplicated_members(score, estimator, backend):
    """Integer ens_w equals the unweighted score of the ensemble with each member
    repeated that many times (holds for the energy form, not for fair or akr)."""
    obs, fct, _, fct_dup, ens_w = _gks_ens_w_inputs(score.endswith("mv_ensemble"))
    res_w = _gks_call(score, estimator, obs, fct, ens_w=ens_w, backend=backend)
    res_dup = _gks_call(score, estimator, obs, fct_dup, backend=backend)
    np.testing.assert_allclose(res_w, res_dup, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("score", ["gksuv_ensemble", "gksmv_ensemble"])
def test_gks_fair_ens_w_known_value(score, backend):
    """Weighted fair estimator excludes the i == j pairs (#150)."""
    mv = score == "gksmv_ensemble"
    obs = np.array([0.3, -0.1]) if mv else 0.3
    fct = (
        np.array([[0.1, 0.2], [-0.4, 0.0], [1.2, -0.5]])
        if mv
        else np.array([0.1, -0.4, 1.2])
    )
    w = np.array([0.5, 0.3, 0.2])
    kern = lambda a, b: np.exp(-0.5 * np.sum((np.asarray(a) - b) ** 2))  # noqa: E731
    e_1 = sum(w[i] * kern(fct[i], obs) for i in range(3))
    e_2 = sum(
        w[i] * w[j] * kern(fct[i], fct[j]) for i in range(3) for j in range(3) if i != j
    )
    expected = -(e_1 - 0.5 * e_2 / (1 - np.sum(w**2)) - 0.5)
    res = getattr(sr, score)(obs, fct, ens_w=w, estimator="fair", backend=backend)
    # rtol for jax, which runs in float32 here
    np.testing.assert_allclose(np.asarray(res), expected, rtol=1e-5)


@pytest.mark.parametrize("score", ["gksuv_ensemble", "gksmv_ensemble"])
def test_gks_akr_circperm_ens_w_known_value(score, backend):
    """Weighted akr_circperm pairs member i with (i + M // 2) % M and uses w_i (#149)."""
    mv = score == "gksmv_ensemble"
    obs = [0.3, -0.1] if mv else [0.3]
    fct = [[0.1, 0.2], [-0.4, 0.0], [1.2, -0.5]] if mv else [[0.1], [-0.4], [1.2]]
    w = [0.5, 0.3, 0.2]
    partner = [1, 2, 0]  # M = 3, M // 2 = 1

    def kern(a, b):
        return math.exp(-0.5 * sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

    e_1 = sum(w[i] * kern(fct[i], obs) for i in range(3))
    e_2 = sum(w[i] * kern(fct[i], fct[partner[i]]) for i in range(3))
    expected = -(e_1 - 0.5 * e_2 - 0.5)
    obs_in = np.array(obs) if mv else obs[0]
    fct_in = np.array(fct) if mv else np.array(fct)[:, 0]
    res = getattr(sr, score)(
        obs_in, fct_in, ens_w=np.array(w), estimator="akr_circperm", backend=backend
    )
    # rtol for jax, which runs in float32 here
    np.testing.assert_allclose(np.asarray(res), expected, rtol=1e-5)
