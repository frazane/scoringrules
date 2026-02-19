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


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_variogram_score_correctness(estimator, backend):
    # correctness
    fct = np.array(
        [
            [0.79546742, 0.4777960, 0.2164079, 0.5409873],
            [0.02461368, 0.7584595, 0.3181810, 0.1319422],
        ]
    ).T
    obs = np.array([0.2743836, 0.8146400])

    res = sr.vs_ensemble(obs, fct, p=0.5, estimator=estimator, backend=backend)
    if estimator == "nrg":
        expected = 0.04114727
        np.testing.assert_allclose(res, expected, rtol=1e-5)
    elif estimator == "fair":
        expected = 0.01407421
        np.testing.assert_allclose(res, expected, rtol=1e-5)

    res = sr.vs_ensemble(obs, fct, p=1.0, estimator=estimator, backend=backend)
    if estimator == "nrg":
        expected = 0.04480374
        np.testing.assert_allclose(res, expected, rtol=1e-5)
    elif estimator == "fair":
        expected = 0.004730382
        np.testing.assert_allclose(res, expected, rtol=1e-5)

    # with and without dimension weights
    w = np.ones((2, 2))
    res_w = sr.vs_ensemble(obs, fct, w=w, p=1.0, estimator=estimator, backend=backend)
    np.testing.assert_allclose(res_w, res, rtol=1e-5)

    # with dimension weights
    w = np.array([[0.1, 0.2], [0.2, 0.4]])
    res = sr.vs_ensemble(obs, fct, w=w, p=0.5, estimator=estimator, backend=backend)
    if estimator == "nrg":
        expected = 0.008229455
        np.testing.assert_allclose(res, expected, atol=1e-6)

    # invariance to diagonal terms of weight matrix
    w = np.array([[1.1, 0.2], [0.2, 100]])
    res = sr.vs_ensemble(obs, fct, w=w, p=0.5, estimator=estimator, backend=backend)
    if estimator == "nrg":
        np.testing.assert_allclose(res, expected, atol=1e-6)

    # with ensemble weights
    ens_w = np.array([0.1, 0.8, 0.3, 0.4])
    res = sr.vs_ensemble(obs, fct, ens_w=ens_w, p=0.5, estimator="nrg", backend=backend)
    if estimator == "nrg":
        expected = 0.07648068
        np.testing.assert_allclose(res, expected, atol=1e-6)

    ens_w = np.array([0.9, 0.2, 0.5, 0.5])
    res = sr.vs_ensemble(
        obs, fct, ens_w=ens_w, p=0.75, estimator=estimator, backend=backend
    )
    if estimator == "nrg":
        expected = 0.01160571
        np.testing.assert_allclose(res, expected, atol=1e-6)

    # check invariance to scaling of the weights
    ens_w_scaled = ens_w * 10
    res_scaled = sr.vs_ensemble(
        obs, fct, ens_w=ens_w_scaled, p=0.75, estimator=estimator, backend=backend
    )
    assert np.allclose(res_scaled, res)

    # with dimension and ensemble weights
    w = np.array([[0.1, 0.2], [0.2, 0.4]])
    ens_w = np.array([0.1, 0.8, 0.3, 0.4])
    res = sr.vs_ensemble(
        obs, fct, w=w, ens_w=ens_w, p=0.5, estimator=estimator, backend=backend
    )
    if estimator == "nrg":
        expected = 0.01529614
        np.testing.assert_allclose(res, expected, atol=1e-6)
