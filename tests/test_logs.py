import numpy as np
import pytest
from scoringrules import _logs

from .conftest import BACKENDS

ENSEMBLE_SIZE = 51
N = 100


@pytest.mark.parametrize("backend", BACKENDS)
def test_laplace(backend):
    obs, location, scale = -3.0, 0.1, 0.9
    res = _logs.logs_laplace(obs, backend=backend)
    expected = 3.693147
    assert np.isclose(res, expected)

    res = _logs.logs_laplace(obs, location, backend=backend)
    expected = 3.793147
    assert np.isclose(res, expected)

    res = _logs.logs_laplace(obs, location, scale, backend=backend)
    expected = 4.032231
    assert np.isclose(res, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_loglaplace(backend):
    obs, locationlog, scalelog = 3.0, 0.1, 0.9
    res = _logs.logs_loglaplace(obs, locationlog, scalelog, backend=backend)
    expected = 2.795968
    assert np.isclose(res, expected)

    obs, locationlog, scalelog = 0.0, 0.1, 0.9
    res = _logs.logs_loglaplace(obs, locationlog, scalelog, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    obs, locationlog, scalelog = 12.0, 12.1, 4.9
    res = _logs.logs_loglaplace(obs, locationlog, scalelog, backend=backend)
    expected = 6.729553
    assert np.isclose(res, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_normal(backend):
    res = _logs.logs_normal(0.0, 0.1, 0.1, backend=backend)
    assert np.isclose(res, -0.8836466, rtol=1e-5)
