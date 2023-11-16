import numpy as np
import pytest

from scoringrules._variogram import (
    owvariogram_score,
    twvariogram_score,
    variogram_score,
    vrvariogram_score,
)
from scoringrules.backend import backends

from .conftest import JAX_IMPORTED, TENSORFLOW_IMPORTED, TORCH_IMPORTED

ENSEMBLE_SIZE = 51
N = 100
N_VARS = 3

BACKENDS = ["numpy", "numba"]
if JAX_IMPORTED:
    BACKENDS.append("jax")
    import jax

    jax.config.update("jax_enable_x64", True)
if TORCH_IMPORTED:
    BACKENDS.append("torch")
if TENSORFLOW_IMPORTED:
    BACKENDS.append("tensorflow")


@pytest.mark.parametrize("backend", BACKENDS)
def test_owvs_vs_vs(backend):
    obs = np.random.randn(N, N_VARS)
    fcts = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)

    res = variogram_score(fcts, obs, backend=backend)
    resw = owvariogram_score(
        fcts,
        obs,
        lambda x: backends[backend].mean(x) * 0.0 + 1.0,
        backend=backend,
    )
    np.testing.assert_allclose(res, resw, rtol=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_twvs_vs_vs(backend):
    obs = np.random.randn(N, N_VARS)
    fcts = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)

    res = variogram_score(fcts, obs, backend=backend)
    resw = twvariogram_score(fcts, obs, lambda x: x, backend=backend)
    np.testing.assert_allclose(res, resw, rtol=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_vrvs_vs_vs(backend):
    obs = np.random.randn(N, N_VARS)
    fcts = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)

    res = variogram_score(fcts, obs, backend=backend)
    resw = vrvariogram_score(
        fcts,
        obs,
        lambda x: backends[backend].mean(x) * 0.0 + 1.0,
        backend=backend,
    )
    np.testing.assert_allclose(res, resw, rtol=5e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_owvariogram_score_correctness(backend):
    fcts = np.array(
        [[0.79546742, 0.4777960, 0.2164079], [0.02461368, 0.7584595, 0.3181810]]
    ).T
    obs = np.array([0.2743836, 0.8146400])

    def w_func(x):
        return (
            backends[backend].all(x > 0.2) + 0.0
        )  # + 0.0 works to convert to float in every backend

    res = owvariogram_score(fcts, obs, w_func, p=0.5, backend=backend)
    np.testing.assert_allclose(res, 0.1929739, rtol=1e-6)

    def w_func(x):
        return backends[backend].all(x < 1.0) + 0.0

    res = owvariogram_score(fcts, obs, w_func, p=1.0, backend=backend)
    np.testing.assert_allclose(res, 0.04856366, rtol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_twvariogram_score_correctness(backend):
    fcts = np.array(
        [[0.79546742, 0.4777960, 0.2164079], [0.02461368, 0.7584595, 0.3181810]]
    ).T
    obs = np.array([0.2743836, 0.8146400])

    def v_func(x):
        return np.maximum(x, 0.2)

    res = twvariogram_score(fcts, obs, v_func, p=0.5, backend=backend)
    np.testing.assert_allclose(res, 0.07594679, rtol=1e-6)

    def v_func(x):
        return np.minimum(x, 1.0)

    res = twvariogram_score(fcts, obs, v_func, p=1.0, backend=backend)
    np.testing.assert_allclose(res, 0.04856366, rtol=1e-6)
