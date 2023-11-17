import numpy as np
import pytest

from scoringrules._crps import (
    crps_ensemble,
    owcrps_ensemble,
    twcrps_ensemble,
    vrcrps_ensemble,
)

from .conftest import BACKENDS

ENSEMBLE_SIZE = 51
N = 100


@pytest.mark.parametrize("backend", BACKENDS)
def test_twcrps_vs_crps(backend):
    obs = np.random.randn(N)
    mu = obs + np.random.randn(N) * 0.1
    sigma = abs(np.random.randn(N)) * 0.3
    fcts = np.random.randn(N, ENSEMBLE_SIZE) * sigma[..., None] + mu[..., None]

    res = crps_ensemble(fcts, obs, backend=backend, estimator="nrg")
    resw = twcrps_ensemble(fcts, obs, lambda x: x, estimator="nrg", backend=backend)
    np.testing.assert_allclose(res, resw, rtol=1e-10)


@pytest.mark.parametrize("backend", BACKENDS)
def test_owcrps_vs_crps(backend):
    obs = np.random.randn(N)
    mu = obs + np.random.randn(N) * 0.1
    sigma = abs(np.random.randn(N)) * 0.5
    fcts = np.random.randn(N, ENSEMBLE_SIZE) * sigma[..., None] + mu[..., None]

    res = crps_ensemble(fcts, obs, backend=backend, estimator="nrg")
    resw = owcrps_ensemble(fcts, obs, lambda x: x * 0.0 + 1.0, backend=backend)
    np.testing.assert_allclose(res, resw, rtol=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_vrcrps_vs_crps(backend):
    obs = np.random.randn(N)
    mu = obs + np.random.randn(N) * 0.1
    sigma = abs(np.random.randn(N)) * 0.3
    fcts = np.random.randn(N, ENSEMBLE_SIZE) * sigma[..., None] + mu[..., None]

    res = crps_ensemble(fcts, obs, backend=backend, estimator="nrg")
    resw = vrcrps_ensemble(fcts, obs, lambda x: x * 0.0 + 1.0, backend=backend)
    np.testing.assert_allclose(res, resw, atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_owcrps_score_correctness(backend):
    fcts = np.array(
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
        return (x > -1).astype(float)

    res = np.mean(np.float64(owcrps_ensemble(fcts, obs, w_func, backend=backend)))
    np.testing.assert_allclose(res, 0.09320807, rtol=1e-6)

    def w_func(x):
        return (x < 1.85).astype(float)

    res = np.mean(np.float64(owcrps_ensemble(fcts, obs, w_func, backend=backend)))
    np.testing.assert_allclose(res, 0.09933139, rtol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_twcrps_score_correctness(backend):
    fcts = np.array(
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
        np.float64(twcrps_ensemble(fcts, obs, v_func, estimator="nrg", backend=backend))
    )
    np.testing.assert_allclose(res, 0.09489662, rtol=1e-6)

    def v_func(x):
        return np.minimum(x, 1.85)

    res = np.mean(
        np.float64(twcrps_ensemble(fcts, obs, v_func, estimator="nrg", backend=backend))
    )
    np.testing.assert_allclose(res, 0.0994809, rtol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_vrcrps_score_correctness(backend):
    fcts = np.array(
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
        return (x > -1).astype(float)

    res = np.mean(np.float64(vrcrps_ensemble(fcts, obs, w_func, backend=backend)))
    np.testing.assert_allclose(res, 0.1003983, rtol=1e-6)

    def w_func(x):
        return (x < 1.85).astype(float)

    res = np.mean(np.float64(vrcrps_ensemble(fcts, obs, w_func, backend=backend)))
    np.testing.assert_allclose(res, 0.1950857, rtol=1e-6)
