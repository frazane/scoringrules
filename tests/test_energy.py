import numpy as np
import pytest
import scoringrules as sr

from .conftest import BACKENDS

ENSEMBLE_SIZE = 11
N = 20
N_VARS = 3

ESTIMATORS = ["nrg", "fair", "akr", "akr_circperm"]


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_energy_score(estimator, backend):
    # test exceptions

    # incorrect dimensions
    with pytest.raises(ValueError):
        obs = np.random.randn(N, N_VARS)
        fct = np.random.randn(N, ENSEMBLE_SIZE, N_VARS - 1)
        sr.es_ensemble(obs, fct, estimator=estimator, backend=backend)

    # undefined estimator
    with pytest.raises(ValueError):
        fct = np.random.randn(N, ENSEMBLE_SIZE, N_VARS)
        est = "undefined_estimator"
        sr.es_ensemble(obs, fct, estimator=est, backend=backend)

    # test output

    # shapes
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)
    res = sr.es_ensemble(obs, fct, estimator=estimator, backend=backend)
    assert res.shape == (N,)

    # correctness
    fct = np.array(
        [
            [0.79546742, 0.4777960, 0.2164079, 0.5409873],
            [0.02461368, 0.7584595, 0.3181810, 0.1319422],
        ]
    ).T
    obs = np.array([0.2743836, 0.8146400])

    if estimator == "nrg":
        res = sr.es_ensemble(obs, fct, estimator=estimator, backend=backend)
        expected = 0.3949793
        np.testing.assert_allclose(res, expected, atol=1e-6)
    elif estimator == "fair":
        res = sr.es_ensemble(obs, fct, estimator=estimator, backend=backend)
        expected = 0.3274585
        np.testing.assert_allclose(res, expected, atol=1e-6)
    elif estimator == "akr":
        res = sr.es_ensemble(obs, fct, estimator=estimator, backend=backend)
        expected = 0.3522818
        np.testing.assert_allclose(res, expected, atol=1e-6)
    elif estimator == "akr_circperm":
        res = sr.es_ensemble(obs, fct, estimator=estimator, backend=backend)
        expected = 0.2778118
        np.testing.assert_allclose(res, expected, atol=1e-6)
