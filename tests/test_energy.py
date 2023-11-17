import jax
import numpy as np
import pytest
from scoringrules._energy import energy_score

from .conftest import JAX_IMPORTED, TORCH_IMPORTED, TENSORFLOW_IMPORTED

ENSEMBLE_SIZE = 51
N = 100
N_VARS = 3

BACKENDS = ["numpy", "numba"]
if JAX_IMPORTED:
    BACKENDS.append("jax")
if TORCH_IMPORTED:
    BACKENDS.append("torch")
if TENSORFLOW_IMPORTED:
    BACKENDS.append("tensorflow")


@pytest.mark.parametrize("backend", BACKENDS)
def test_energy_score(backend):
    obs = np.random.randn(N, N_VARS)
    fcts = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)

    res = energy_score(fcts, obs, backend=backend)

    if backend in ["numpy", "numba"]:
        assert isinstance(res, np.ndarray)
    elif backend == "jax":
        assert isinstance(res, jax.Array)
