import numpy as np
import pytest
from scoringrules import _logs

ENSEMBLE_SIZE = 51
N = 100

BACKENDS = ["numpy", "numba", "jax"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_normal(backend):
    res = _logs.logs_normal(0.1, 0.1, 0.0, backend=backend)
    assert np.isclose(res, -0.8836466, rtol=1e-5)
