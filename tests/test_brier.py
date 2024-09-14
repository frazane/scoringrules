import numpy as np
import pytest
from scoringrules import _brier

from .conftest import BACKENDS


@pytest.mark.parametrize("backend", BACKENDS)
def test_brier(backend):
    # test correctness
    obs, fct = 0, 0.271
    res = _brier.brier_score(obs, fct, backend=backend)
    expected = 0.073441
    assert np.isclose(res, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_rps(backend):
    # test correctness
    obs, fct = 3, [0.01, 0.32, 0.44, 0.23]
    res = _brier.rps_score(obs, fct, backend=backend)
    expected = 0.1619
    assert np.isclose(res, expected)

    # test correctness
    obs = [1, 3, 4, 2]
    fct = [
        [0.01, 0.32, 0.44, 0.23],
        [0.12, 0.05, 0.48, 0.35],
        [0.09, 0.21, 0.05, 0.65],
        [0.57, 0.31, 0.08, 0.04],
    ]
    res1 = _brier.rps_score(obs, fct, backend=backend)

    fct = np.transpose(fct)
    res2 = _brier.rps_score(obs, fct, axis=0, backend=backend)

    assert np.allclose(res1, res2)
