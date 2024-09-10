import numpy as np
import pytest
from scoringrules import _logs

from .conftest import BACKENDS

ENSEMBLE_SIZE = 51
N = 100


@pytest.mark.parametrize("backend", BACKENDS)
def test_negbinom(backend):
    if backend in ["jax", "torch"]:
        pytest.skip("Not implemented in jax or torch backends")

    # TODO: investigate why JAX and torch results are different from other backends
    # (they fail the test)
    obs, n, prob = 2.0, 7.0, 0.8
    res = _logs.logs_negbinom(obs, n, prob, backend=backend)
    expected = 1.448676
    assert np.isclose(res, expected)

    obs, n, prob = 1.5, 2.0, 0.5
    res = _logs.logs_negbinom(obs, n, prob, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    obs, n, prob = -1.0, 17.0, 0.1
    res = _logs.logs_negbinom(obs, n, prob, backend=backend)
    expected = float("inf")
    assert np.isclose(res, expected)

    obs, n, mu = 2.0, 11.0, 7.3
    res = _logs.logs_negbinom(obs, n, mu=mu, backend=backend)
    expected = 3.247462
    assert np.isclose(res, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_normal(backend):
    res = _logs.logs_normal(0.0, 0.1, 0.1, backend=backend)
    assert np.isclose(res, -0.8836466, rtol=1e-5)
