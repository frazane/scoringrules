import numpy as np
import pytest
import scoringrules as sr


ENSEMBLE_SIZE = 11
N = 20
N_VARS = 3

ESTIMATORS = ["nrg", "fair", "akr", "akr_circperm"]


@pytest.mark.parametrize("estimator", ESTIMATORS)
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

    # test equivalence with and without weights
    w = np.ones(fct.shape[:-1])
    res_nrg_w = sr.es_ensemble(obs, fct, ens_w=w, estimator=estimator, backend=backend)
    res_nrg_now = sr.es_ensemble(obs, fct, estimator=estimator, backend=backend)
    assert np.allclose(res_nrg_w, res_nrg_now)

    # correctness
    fct = np.array(
        [
            [0.79546742, 0.4777960, 0.2164079, 0.5409873],
            [0.02461368, 0.7584595, 0.3181810, 0.1319422],
        ]
    ).T
    obs = np.array([0.2743836, 0.8146400])

    res = sr.es_ensemble(obs, fct, estimator=estimator, backend=backend)
    if estimator == "nrg":
        expected = 0.3949793
        np.testing.assert_allclose(res, expected, atol=1e-6)
    elif estimator == "fair":
        expected = 0.3274585
        np.testing.assert_allclose(res, expected, atol=1e-6)
    elif estimator == "akr":
        expected = 0.3522818
        np.testing.assert_allclose(res, expected, atol=1e-6)
    elif estimator == "akr_circperm":
        expected = 0.2778118
        np.testing.assert_allclose(res, expected, atol=1e-6)

    # correctness with weights
    w = np.array([0.1, 0.2, 0.3, 0.4])

    res = sr.es_ensemble(obs, fct, ens_w=w, estimator=estimator, backend=backend)
    if estimator == "nrg":
        expected = 0.4074382
        np.testing.assert_allclose(res, expected, atol=1e-6)
    elif estimator == "fair":
        expected = 0.333500969
        np.testing.assert_allclose(res, expected, atol=1e-6)
    elif estimator == "akr":
        expected = 0.3345371
        np.testing.assert_allclose(res, expected, atol=1e-6)
    elif estimator == "akr_circperm":
        expected = 0.2612048
        np.testing.assert_allclose(res, expected, atol=1e-6)

    # check invariance to scaling of the weights
    w_scaled = w * 10
    res_scaled = sr.es_ensemble(
        obs, fct, ens_w=w_scaled, estimator=estimator, backend=backend
    )
    np.testing.assert_allclose(res_scaled, res, atol=1e-6)
