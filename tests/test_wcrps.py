import numpy as np
import pytest
import scoringrules as sr

from .conftest import BACKENDS

M = 11
N = 20


@pytest.mark.parametrize("backend", BACKENDS)
def test_owcrps_ensemble(backend):
    # test shapes
    obs = np.random.randn(N)
    res = sr.owcrps_ensemble(obs, np.random.randn(N, M), w_func=lambda x: x * 0.0 + 1.0)
    assert res.shape == (N,)
    res = sr.owcrps_ensemble(
        obs, np.random.randn(M, N), w_func=lambda x: x * 0.0 + 1.0, m_axis=0
    )
    assert res.shape == (N,)


@pytest.mark.parametrize("backend", BACKENDS)
def test_vrcrps_ensemble(backend):
    # test shapes
    obs = np.random.randn(N)
    res = sr.vrcrps_ensemble(obs, np.random.randn(N, M), w_func=lambda x: x * 0.0 + 1.0)
    assert res.shape == (N,)
    res = sr.vrcrps_ensemble(
        obs, np.random.randn(M, N), w_func=lambda x: x * 0.0 + 1.0, m_axis=0
    )
    assert res.shape == (N,)


@pytest.mark.parametrize("backend", BACKENDS)
def test_twcrps_vs_crps(backend):
    obs = np.random.randn(N)
    mu = obs + np.random.randn(N) * 0.1
    sigma = abs(np.random.randn(N)) * 0.3
    fct = np.random.randn(N, M) * sigma[..., None] + mu[..., None]

    res = sr.crps_ensemble(obs, fct, backend=backend, estimator="nrg")

    # no argument given
    resw = sr.twcrps_ensemble(obs, fct, estimator="nrg", backend=backend)
    np.testing.assert_allclose(res, resw, rtol=1e-10)

    # a and b
    resw = sr.twcrps_ensemble(
        obs, fct, a=float("-inf"), b=float("inf"), estimator="nrg", backend=backend
    )
    np.testing.assert_allclose(res, resw, rtol=1e-10)

    # v_func as identity function
    resw = sr.twcrps_ensemble(
        obs, fct, v_func=lambda x: x, estimator="nrg", backend=backend
    )
    np.testing.assert_allclose(res, resw, rtol=1e-10)


@pytest.mark.parametrize("backend", BACKENDS)
def test_owcrps_vs_crps(backend):
    obs = np.random.randn(N)
    mu = obs + np.random.randn(N) * 0.1
    sigma = abs(np.random.randn(N)) * 0.5
    fct = np.random.randn(N, M) * sigma[..., None] + mu[..., None]

    res = sr.crps_ensemble(obs, fct, backend=backend)

    # no argument given
    resw = sr.owcrps_ensemble(obs, fct, backend=backend)
    np.testing.assert_allclose(res, resw, rtol=1e-5)

    # a and b
    resw = sr.owcrps_ensemble(
        obs, fct, a=float("-inf"), b=float("inf"), backend=backend
    )
    np.testing.assert_allclose(res, resw, rtol=1e-5)

    # w_func as identity function
    resw = sr.owcrps_ensemble(obs, fct, w_func=lambda x: x * 0.0 + 1.0, backend=backend)
    np.testing.assert_allclose(res, resw, rtol=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_vrcrps_vs_crps(backend):
    obs = np.random.randn(N)
    mu = obs + np.random.randn(N) * 0.1
    sigma = abs(np.random.randn(N)) * 0.3
    fct = np.random.randn(N, M) * sigma[..., None] + mu[..., None]

    res = sr.crps_ensemble(obs, fct, backend=backend, estimator="nrg")

    # no argument given
    resw = sr.vrcrps_ensemble(obs, fct, backend=backend)
    np.testing.assert_allclose(res, resw, rtol=1e-5)

    # a and b
    resw = sr.vrcrps_ensemble(
        obs, fct, a=float("-inf"), b=float("inf"), backend=backend
    )
    np.testing.assert_allclose(res, resw, rtol=1e-5)

    # w_func as identity function
    resw = sr.vrcrps_ensemble(obs, fct, w_func=lambda x: x * 0.0 + 1.0, backend=backend)
    np.testing.assert_allclose(res, resw, rtol=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_owcrps_score_correctness(backend):
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
        np.float64(sr.owcrps_ensemble(obs, fct, w_func=w_func, backend=backend))
    )
    np.testing.assert_allclose(res, 0.09320807, rtol=1e-6)

    res = np.mean(np.float64(sr.owcrps_ensemble(obs, fct, a=-1.0, backend=backend)))
    np.testing.assert_allclose(res, 0.09320807, rtol=1e-6)

    def w_func(x):
        return (x < 1.85) * 1.0

    res = np.mean(
        np.float64(sr.owcrps_ensemble(obs, fct, w_func=w_func, backend=backend))
    )
    np.testing.assert_allclose(res, 0.09933139, rtol=1e-6)

    res = np.mean(np.float64(sr.owcrps_ensemble(obs, fct, b=1.85, backend=backend)))
    np.testing.assert_allclose(res, 0.09933139, rtol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_twcrps_score_correctness(backend):
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

    def v_func(x):
        return np.maximum(x, -1.0)

    res = np.mean(
        np.float64(
            sr.twcrps_ensemble(
                obs, fct, v_func=v_func, estimator="nrg", backend=backend
            )
        )
    )
    np.testing.assert_allclose(res, 0.09489662, rtol=1e-6)

    res = np.mean(
        np.float64(
            sr.twcrps_ensemble(obs, fct, a=-1.0, estimator="nrg", backend=backend)
        )
    )
    np.testing.assert_allclose(res, 0.09489662, rtol=1e-6)

    def v_func(x):
        return np.minimum(x, 1.85)

    res = np.mean(
        np.float64(
            sr.twcrps_ensemble(
                obs, fct, v_func=v_func, estimator="nrg", backend=backend
            )
        )
    )
    np.testing.assert_allclose(res, 0.0994809, rtol=1e-6)

    res = np.mean(
        np.float64(
            sr.twcrps_ensemble(obs, fct, b=1.85, estimator="nrg", backend=backend)
        )
    )
    np.testing.assert_allclose(res, 0.0994809, rtol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_vrcrps_score_correctness(backend):
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
        np.float64(sr.vrcrps_ensemble(obs, fct, w_func=w_func, backend=backend))
    )
    np.testing.assert_allclose(res, 0.1003983, rtol=1e-6)

    res = np.mean(np.float64(sr.vrcrps_ensemble(obs, fct, a=-1.0, backend=backend)))
    np.testing.assert_allclose(res, 0.1003983, rtol=1e-6)

    def w_func(x):
        return (x < 1.85) * 1.0

    res = np.mean(
        np.float64(sr.vrcrps_ensemble(obs, fct, w_func=w_func, backend=backend))
    )
    np.testing.assert_allclose(res, 0.1950857, rtol=1e-6)

    res = np.mean(np.float64(sr.vrcrps_ensemble(obs, fct, b=1.85, backend=backend)))
    np.testing.assert_allclose(res, 0.1950857, rtol=1e-6)
