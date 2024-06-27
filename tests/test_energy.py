import jax
import numpy as np
import pytest
from scoringrules._energy import energy_score, energy_score_iid, energy_score_kband

from .conftest import BACKENDS

ENSEMBLE_SIZE = 51
N = 100
N_VARS = 3


@pytest.mark.parametrize("backend", BACKENDS)
def test_energy_score(backend):
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)

    res = energy_score(obs, fct, backend=backend)

    if backend in ["numpy", "numba"]:
        assert isinstance(res, np.ndarray)
    elif backend == "jax":
        assert isinstance(res, jax.Array)


@pytest.mark.parametrize("backend", BACKENDS)
def test_energy_score_iid(backend):
    M = 1000
    D = 5
    N = 100

    generator = np.random.default_rng()
    fct = generator.normal(0, 1, (N, M, D))
    obs = np.zeros((N, D))
    res_iid = energy_score_iid(obs, fct)
    res_full = energy_score(obs, fct)
    assert np.all(1 - res_iid / res_full < 0.10), "Error is to full really big"


@pytest.mark.parametrize("backend", BACKENDS)
def test_energy_score_kband(backend):
    M = 1000
    D = 5
    N = 100
    K = 100

    generator = np.random.default_rng()
    fct = generator.normal(0, 1, (N, M, D))
    obs = np.zeros((N, D))
    res_iid = energy_score_kband(obs, fct, K)
    res_full = energy_score(obs, fct)
    assert np.all(1 - res_iid / res_full < 0.05), "Error is to full really big"
