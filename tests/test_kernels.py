import jax
import numpy as np
import pytest
from scoringrules import _kernels

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

    if estimator == "nrg":
        # approx zero when perfect forecast
        perfect_fct = obs[..., None] + np.random.randn(N, ENSEMBLE_SIZE) * 0.00001
        res = _kernels.gksuv_ensemble(
            obs, perfect_fct, estimator=estimator, backend=backend
        )
        res = np.asarray(res)
        assert not np.any(res - 0.0 > 0.0001)

        # test correctness
        obs, fct = 11.6, np.array([9.8, 8.7, 11.9, 12.1, 13.4])
        res = _kernels.gksuv_ensemble(obs, fct, estimator=estimator, backend=backend)
        expected = 0.2490516
        assert np.isclose(res, expected)

    elif estimator == "fair":
        # test correctness
        obs, fct = 11.6, np.array([9.8, 8.7, 11.9, 12.1, 13.4])
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
