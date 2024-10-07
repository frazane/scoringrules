import jax
import numpy as np
import pytest
from scoringrules import _kernels
from scoringrules.backend import backends

from .conftest import BACKENDS

ENSEMBLE_SIZE = 51
N = 100
N_VARS = 3

ESTIMATORS = ["nrg", "fair"]


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_gksuv(estimator, backend):
    obs = np.random.randn(N)
    mu = obs + np.random.randn(N) * 0.1
    sigma = abs(np.random.randn(N)) * 0.3
    fct = np.random.randn(N, ENSEMBLE_SIZE) * sigma[..., None] + mu[..., None]

    # non-negative values
    res = _kernels.gksuv_ensemble(obs, fct, estimator=estimator, backend=backend)
    res = np.asarray(res)
    assert not np.any(res < 0.0)

    obs, fct = 11.6, np.array([9.8, 8.7, 11.9, 12.1, 13.4])
    if estimator == "nrg":
        # approx zero when perfect forecast
        perfect_fct = obs[..., None] + np.random.randn(N, ENSEMBLE_SIZE) * 0.00001
        res = _kernels.gksuv_ensemble(
            obs, perfect_fct, estimator=estimator, backend=backend
        )
        res = np.asarray(res)
        assert not np.any(res - 0.0 > 0.0001)

        # test correctness
        res = _kernels.gksuv_ensemble(obs, fct, estimator=estimator, backend=backend)
        expected = 0.2490516
        assert np.isclose(res, expected)

    elif estimator == "fair":
        # test correctness
        res = _kernels.gksuv_ensemble(obs, fct, estimator=estimator, backend=backend)
        expected = 0.2987752
        assert np.isclose(res, expected)


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_gksmv(estimator, backend):
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)

    res = _kernels.gksmv_ensemble(obs, fct, estimator=estimator, backend=backend)

    if backend in ["numpy", "numba"]:
        assert isinstance(res, np.ndarray)
    elif backend == "jax":
        assert isinstance(res, jax.Array)

    if estimator == "nrg":
        # approx zero when perfect forecast
        perfect_fct = (
            np.expand_dims(obs, axis=-2)
            + np.random.randn(N, ENSEMBLE_SIZE, N_VARS) * 0.00001
        )
        res = _kernels.gksmv_ensemble(
            obs, perfect_fct, estimator=estimator, backend=backend
        )
        res = np.asarray(res)
        assert not np.any(res - 0.0 > 0.0001)

        # test correctness
        obs = np.array([11.6, -23.1])
        fct = np.array(
            [[9.8, 8.7, 11.9, 12.1, 13.4], [-24.8, -18.5, -29.9, -18.3, -21.0]]
        ).transpose()
        res = _kernels.gksmv_ensemble(obs, fct, estimator=estimator, backend=backend)
        expected = 0.5868737
        assert np.isclose(res, expected)

    elif estimator == "fair":
        # test correctness
        obs = np.array([11.6, -23.1])
        fct = np.array(
            [[9.8, 8.7, 11.9, 12.1, 13.4], [-24.8, -18.5, -29.9, -18.3, -21.0]]
        ).transpose()
        res = _kernels.gksmv_ensemble(obs, fct, estimator=estimator, backend=backend)
        expected = 0.6120162
        assert np.isclose(res, expected)


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_twgksuv(estimator, backend):
    obs = np.random.randn(N)
    mu = obs + np.random.randn(N) * 0.1
    sigma = abs(np.random.randn(N)) * 0.3
    fct = np.random.randn(N, ENSEMBLE_SIZE) * sigma[..., None] + mu[..., None]

    res = _kernels.gksuv_ensemble(obs, fct, estimator=estimator, backend=backend)
    resw = _kernels.twgksuv_ensemble(
        obs, fct, lambda x: x, estimator=estimator, backend=backend
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
                _kernels.twgksuv_ensemble(
                    obs, fct, v_func1, estimator=estimator, backend=backend
                )
            )
        )
        np.testing.assert_allclose(res, 0.01018774, rtol=1e-6)

        res = np.mean(
            np.float64(
                _kernels.twgksuv_ensemble(
                    obs, fct, v_func2, estimator=estimator, backend=backend
                )
            )
        )
        np.testing.assert_allclose(res, 0.0089314, rtol=1e-6)

    elif estimator == "fair":
        res = np.mean(
            np.float64(
                _kernels.twgksuv_ensemble(
                    obs, fct, v_func1, estimator=estimator, backend=backend
                )
            )
        )
        np.testing.assert_allclose(res, 0.130842, rtol=1e-6)

        res = np.mean(
            np.float64(
                _kernels.twgksuv_ensemble(
                    obs, fct, v_func2, estimator=estimator, backend=backend
                )
            )
        )
        np.testing.assert_allclose(res, 0.1283745, rtol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_twgksmv(backend):
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)

    res0 = _kernels.gksmv_ensemble(obs, fct, backend=backend)
    res = _kernels.twgksmv_ensemble(obs, fct, lambda x: x, backend=backend)
    np.testing.assert_allclose(res, res0, rtol=1e-6)

    fct = np.array(
        [[0.79546742, 0.4777960, 0.2164079], [0.02461368, 0.7584595, 0.3181810]]
    ).T
    obs = np.array([0.2743836, 0.8146400])

    def v_func(x):
        return np.maximum(x, 0.2)

    res = _kernels.twgksmv_ensemble(obs, fct, v_func, backend=backend)
    np.testing.assert_allclose(res, 0.08671498, rtol=1e-6)

    def v_func(x):
        return np.minimum(x, 1)

    res = _kernels.twgksmv_ensemble(obs, fct, v_func, backend=backend)
    np.testing.assert_allclose(res, 0.1016436, rtol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_owgksuv(backend):
    obs = np.random.randn(N)
    mu = obs + np.random.randn(N) * 0.1
    sigma = abs(np.random.randn(N)) * 0.3
    fct = np.random.randn(N, ENSEMBLE_SIZE) * sigma[..., None] + mu[..., None]

    res = _kernels.gksuv_ensemble(obs, fct, backend=backend)
    resw = _kernels.owgksuv_ensemble(obs, fct, lambda x: x * 0.0 + 1.0, backend=backend)
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

    def w_func(x):
        return (x > -1).astype(float)

    res = np.mean(
        np.float64(_kernels.owgksuv_ensemble(obs, fct, w_func, backend=backend))
    )
    np.testing.assert_allclose(res, 0.01036335, rtol=1e-6)

    def w_func(x):
        return (x < 1.85).astype(float)

    res = np.mean(
        np.float64(_kernels.owgksuv_ensemble(obs, fct, w_func, backend=backend))
    )
    np.testing.assert_allclose(res, 0.008905213, rtol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_owgksmv(backend):
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)

    res0 = _kernels.gksmv_ensemble(obs, fct, backend=backend)
    res = _kernels.owgksmv_ensemble(
        obs, fct, lambda x: backends[backend].mean(x) * 0.0 + 1.0, backend=backend
    )
    np.testing.assert_allclose(res, res0, rtol=1e-6)

    fct = np.array(
        [[0.79546742, 0.4777960, 0.2164079], [0.02461368, 0.7584595, 0.3181810]]
    ).T
    obs = np.array([0.2743836, 0.8146400])

    def w_func(x):
        return backends[backend].all(x > 0.2)

    res = _kernels.owgksmv_ensemble(obs, fct, w_func, backend=backend)
    np.testing.assert_allclose(res, 0.03901075, rtol=1e-6)

    def w_func(x):
        return backends[backend].all(x < 1.0)

    res = _kernels.owgksmv_ensemble(obs, fct, w_func, backend=backend)
    np.testing.assert_allclose(res, 0.1016436, rtol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_vrgksuv(backend):
    obs = np.random.randn(N)
    mu = obs + np.random.randn(N) * 0.1
    sigma = abs(np.random.randn(N)) * 0.3
    fct = np.random.randn(N, ENSEMBLE_SIZE) * sigma[..., None] + mu[..., None]

    res = _kernels.gksuv_ensemble(obs, fct, backend=backend)
    resw = _kernels.vrgksuv_ensemble(obs, fct, lambda x: x * 0.0 + 1.0, backend=backend)
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

    def w_func(x):
        return (x > -1).astype(float)

    res = np.mean(
        np.float64(_kernels.vrgksuv_ensemble(obs, fct, w_func, backend=backend))
    )
    np.testing.assert_allclose(res, 0.01476682, rtol=1e-6)

    def w_func(x):
        return (x < 1.85).astype(float)

    res = np.mean(
        np.float64(_kernels.vrgksuv_ensemble(obs, fct, w_func, backend=backend))
    )
    np.testing.assert_allclose(res, 0.04011836, rtol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_vrgksmv(backend):
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)

    res0 = _kernels.gksmv_ensemble(obs, fct, backend=backend)
    res = _kernels.vrgksmv_ensemble(
        obs, fct, lambda x: backends[backend].mean(x) * 0.0 + 1.0, backend=backend
    )
    np.testing.assert_allclose(res, res0, rtol=1e-6)
