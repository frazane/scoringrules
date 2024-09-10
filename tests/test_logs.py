import numpy as np
import pytest
from scoringrules import _logs

from .conftest import BACKENDS

ENSEMBLE_SIZE = 51
N = 100


@pytest.mark.parametrize("backend", BACKENDS)
def test_tlogis(backend):
    obs, location, scale, lower, upper = 4.9, 3.5, 2.3, 0.0, 20.0
    res = _logs.logs_tlogistic(obs, location, scale, lower, upper, backend=backend)
    expected = 2.11202
    assert np.isclose(res, expected)

    # aligns with logs_logistic
    # res0 = _logs.logs_logistic(obs, location, scale, backend=backend)
    # res = _logs.logs_tlogistic(obs, location, scale, backend=backend)
    # assert np.isclose(res, res0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_tnormal(backend):
    obs, location, scale, lower, upper = 4.2, 2.9, 2.2, 1.5, 17.3
    res = _logs.logs_tnormal(obs, location, scale, lower, upper, backend=backend)
    expected = 1.577806
    assert np.isclose(res, expected)

    obs, location, scale, lower, upper = -1.0, 2.9, 2.2, 1.5, 17.3
    res = _logs.logs_tnormal(obs, location, scale, lower, upper, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    # aligns with logs_normal
    res0 = _logs.logs_normal(obs, location, scale, backend=backend)
    res = _logs.logs_tnormal(obs, location, scale, backend=backend)
    assert np.isclose(res, res0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_tt(backend):
    if backend in ["jax", "torch", "tensorflow"]:
        pytest.skip("Not implemented in jax, torch or tensorflow backends")

    obs, df, location, scale, lower, upper = 1.9, 2.9, 3.1, 4.2, 1.5, 17.3
    res = _logs.logs_tt(obs, df, location, scale, lower, upper, backend=backend)
    expected = 2.002856
    assert np.isclose(res, expected)

    obs, df, location, scale, lower, upper = -1.0, 2.9, 3.1, 4.2, 1.5, 17.3
    res = _logs.logs_tt(obs, df, location, scale, lower, upper, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    # aligns with logs_t
    # res0 = _logs.logs_t(obs, df, location, scale, backend=backend)
    # res = _logs.logs_tt(obs, df, location, scale, backend=backend)
    # assert np.isclose(res, res0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_normal(backend):
    res = _logs.logs_normal(0.0, 0.1, 0.1, backend=backend)
    assert np.isclose(res, -0.8836466, rtol=1e-5)
