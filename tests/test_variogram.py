import jax
import numpy as np
import pytest
from scoringrules._variogram import variogram_score

from .conftest import BACKENDS

ENSEMBLE_SIZE = 51
N = 100
N_VARS = 30


@pytest.mark.parametrize("backend", BACKENDS)
def test_variogram_score(backend):
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)

    res = variogram_score(obs, fct, backend=backend)

    if backend in ["numpy", "numba"]:
        assert isinstance(res, np.ndarray)
    elif backend == "jax":
        assert isinstance(res, jax.Array)


@pytest.mark.parametrize("backend", BACKENDS)
def test_variogram_score_permuted_dims(backend):
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)

    res = variogram_score(obs, fct, v_axis=-1, m_axis=-2, backend=backend)

    if backend in ["numpy", "numba"]:
        assert isinstance(res, np.ndarray)
    elif backend == "jax":
        assert isinstance(res, jax.Array)


@pytest.mark.parametrize("backend", BACKENDS)
def test_variogram_score_correctness(backend):
    fct = np.array(
        [[0.79546742, 0.4777960, 0.2164079], [0.02461368, 0.7584595, 0.3181810]]
    )

    obs = np.array([0.2743836, 0.8146400])

    res = variogram_score(obs, fct.T, p=0.5, backend=backend)
    np.testing.assert_allclose(res, 0.05083489, rtol=1e-5)

    res = variogram_score(obs, fct.T, p=1.0, backend=backend)
    np.testing.assert_allclose(res, 0.04856365, rtol=1e-5)
