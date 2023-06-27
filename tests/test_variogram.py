import jax
import numpy as np
import pytest
from scoringrules._variogram import variogram_score

ENSEMBLE_SIZE = 51
N = 100
N_VARS = 30

BACKENDS = ["numpy", "numba", "jax"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_variogram_score(backend):
    obs = np.random.randn(N, N_VARS)
    fcts = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)

    res = variogram_score(fcts, obs, backend=backend)

    if backend in ["numpy", "numba"]:
        assert isinstance(res, np.ndarray)
    elif backend == "jax":
        assert isinstance(res, jax.Array)
