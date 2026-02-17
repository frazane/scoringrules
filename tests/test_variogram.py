import numpy as np
import pytest

import scoringrules as sr


ENSEMBLE_SIZE = 51
N = 100
N_VARS = 30

ESTIMATORS = ["nrg", "fair"]


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_variogram_score(estimator, backend):
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)

    res = sr.vs_ensemble(obs, fct, estimator=estimator, backend=backend)

    if backend in ["numpy", "numba"]:
        assert isinstance(res, np.ndarray)
    elif backend == "jax":
        assert "jax" in res.__module__


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_variogram_score_permuted_dims(estimator, backend):
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)

    res = sr.vs_ensemble(
        obs, fct, v_axis=-1, m_axis=-2, estimator=estimator, backend=backend
    )

    if backend in ["numpy", "numba"]:
        assert isinstance(res, np.ndarray)
    elif backend == "jax":
        assert "jax" in res.__module__


def test_variogram_score_correctness(backend):
    fct = np.array(
        [
            [0.79546742, 0.4777960, 0.2164079, 0.5409873],
            [0.02461368, 0.7584595, 0.3181810, 0.1319422],
        ]
    ).T
    obs = np.array([0.2743836, 0.8146400])

    res = sr.vs_ensemble(obs, fct, p=0.5, estimator="nrg", backend=backend)
    np.testing.assert_allclose(res, 0.04114727, rtol=1e-5)

    res = sr.vs_ensemble(obs, fct, p=0.5, estimator="fair", backend=backend)
    np.testing.assert_allclose(res, 0.01407421, rtol=1e-5)

    res = sr.vs_ensemble(obs, fct, p=1.0, estimator="nrg", backend=backend)
    np.testing.assert_allclose(res, 0.04480374, rtol=1e-5)

    res = sr.vs_ensemble(obs, fct, p=1.0, estimator="fair", backend=backend)
    np.testing.assert_allclose(res, 0.004730382, rtol=1e-5)
