import numpy as np
import pytest
from scoringrules._variogram import variogram_score
from scoringrules._wvariogram import (
    owvariogram_score,
    twvariogram_score,
    vrvariogram_score,
)

from .conftest import JAX_IMPORTED

ENSEMBLE_SIZE = 51
N = 100
N_VARS = 3

BACKENDS = ["numpy", "numba"]
if JAX_IMPORTED:
    BACKENDS.append("jax")
    import jax

    jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("backend", BACKENDS)
def test_owvs_vs_vs(backend):
    obs = np.random.randn(N, N_VARS)
    fcts = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)

    res = variogram_score(fcts, obs, backend=backend)
    resw = owvariogram_score(fcts, obs, backend=backend)
    np.testing.assert_allclose(res, resw, rtol=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_twvs_vs_vs(backend):
    obs = np.random.randn(N, N_VARS)
    fcts = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)

    res = variogram_score(fcts, obs, backend=backend)
    resw = twvariogram_score(fcts, obs, backend=backend)
    np.testing.assert_allclose(res, resw, rtol=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_vrvs_vs_vs(backend):
    obs = np.random.randn(N, N_VARS)
    fcts = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)

    res = variogram_score(fcts, obs, backend=backend)
    resw = vrvariogram_score(fcts, obs, backend=backend)
    np.testing.assert_allclose(res, resw, rtol=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_owvariogram_score_correctness(backend):
    fcts = np.array(
        [[0.79546742, 0.4777960, 0.2164079], [0.02461368, 0.7584595, 0.3181810]]
    ).T
    obs = np.array([0.2743836, 0.8146400])

    def w_func(x, t):
        return (x > t).all().astype(np.float32)

    t = 0.2
    res = owvariogram_score(fcts, obs, 0.5, w_func, (t,), backend=backend)
    np.testing.assert_allclose(res, 0.1929739, rtol=1e-6)

    def w_func(x, t):
        return (x < t).all().astype(np.float32)

    t = 1
    res = owvariogram_score(fcts, obs, 1, w_func, (t,), backend=backend)
    np.testing.assert_allclose(res, 0.04856366, rtol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_twvariogram_score_correctness(backend):
    fcts = np.array(
        [[0.79546742, 0.4777960, 0.2164079], [0.02461368, 0.7584595, 0.3181810]]
    ).T
    obs = np.array([0.2743836, 0.8146400])

    def v_func(x, t):
        return np.maximum(x, t)

    t = 0.2
    res = twvariogram_score(fcts, obs, 0.5, v_func, (t,), backend=backend)
    np.testing.assert_allclose(res, 0.07594679, rtol=1e-6)

    def v_func(x, t):
        return np.minimum(x, t)

    t = 1
    res = twvariogram_score(fcts, obs, 1, v_func, (t,), backend=backend)
    np.testing.assert_allclose(res, 0.04856366, rtol=1e-6)
