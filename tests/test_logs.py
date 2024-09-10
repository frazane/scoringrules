import numpy as np
import pytest
from scoringrules import _logs

from .conftest import BACKENDS

ENSEMBLE_SIZE = 51
N = 100


@pytest.mark.parametrize("backend", BACKENDS)
def test_beta(backend):
    res = _logs.logs_beta(
        np.random.uniform(0, 1, (3, 3)),
        np.random.uniform(0, 3, (3, 3)),
        1.1,
        backend=backend,
    )
    assert res.shape == (3, 3)
    assert not np.any(np.isnan(res))

    obs, shape1, shape2, lower, upper = 0.3, 0.7, 1.1, 0.0, 1.0
    res = _logs.logs_beta(obs, shape1, shape2, lower, upper, backend=backend)
    expected = -0.04344567
    assert np.isclose(res, expected)

    obs, shape1, shape2, lower, upper = -3.0, 0.9, 2.1, -5.0, 4.0
    res = _logs.logs_beta(obs, shape1, shape2, lower, upper, backend=backend)
    expected = 1.74193
    assert np.isclose(res, expected)

    obs, shape1, shape2, lower, upper = -3.0, 0.9, 2.1, -2.0, 4.0
    res = _logs.logs_beta(obs, shape1, shape2, lower, upper, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_exponential2(backend):
    obs, location, scale = 8.3, 7.0, 1.2
    res = _logs.logs_exponential2(obs, location, scale, backend=backend)
    expected = 1.265655
    assert np.isclose(res, expected)

    obs, location, scale = -1.3, 2.4, 4.5
    res = _logs.logs_exponential2(obs, location, scale, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    obs, location, scale = 0.9, 0.9, 2.0
    res = _logs.logs_exponential2(obs, location, scale, backend=backend)
    expected = 0.6931472
    assert np.isclose(res, expected)

    # res0 = _logs.logs_exponential(obs, 1 / scale, backend=backend)
    # res = _logs.logs_exponential2(obs, 0.0, scale, backend=backend)
    # assert np.isclose(res, res0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_2pexponential(backend):
    obs, scale1, scale2, location = 0.3, 0.1, 4.3, 0.0
    res = _logs.logs_2pexponential(obs, scale1, scale2, location, backend=backend)
    expected = 1.551372
    assert np.isclose(res, expected)

    obs, scale1, scale2, location = -20.8, 7.1, 2.0, -15.4
    res = _logs.logs_2pexponential(obs, scale1, scale2, location, backend=backend)
    expected = 2.968838
    assert np.isclose(res, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_normal(backend):
    res = _logs.logs_normal(0.0, 0.1, 0.1, backend=backend)
    assert np.isclose(res, -0.8836466, rtol=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_2pnormal(backend):
    obs, scale1, scale2, location = 29.1, 4.6, 1.3, 27.9
    res = _logs.logs_2pnormal(obs, scale1, scale2, location, backend=backend)
    expected = 2.426779
    assert np.isclose(res, expected)

    obs, scale1, scale2, location = -2.2, 1.6, 3.3, -1.9
    res = _logs.logs_2pnormal(obs, scale1, scale2, location, backend=backend)
    expected = 1.832605
    assert np.isclose(res, expected)

    obs, scale, location = 1.5, 4.5, 5.4
    res0 = _logs.logs_normal(obs, location, scale, backend=backend)
    res = _logs.logs_2pnormal(obs, scale, scale, location, backend=backend)
    assert np.isclose(res, res0)
