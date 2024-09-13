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
    obs, fct = np.array([0, 0, 1, 0]), np.array([0.01, 0.32, 0.44, 0.23])
    res = _brier.rps_score(obs, fct, backend=backend)
    expected = 0.1619
    assert np.isclose(res, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_logs(backend):
    # test correctness
    obs, fct = 0, 0.481
    res = _brier.log_score(obs, fct, backend=backend)
    expected = 0.6558514
    assert np.isclose(res, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_rls(backend):
    # test correctness
    obs, fct = np.array([0, 1, 0, 0]), np.array([0.21, 0.31, 0.32, 0.16])
    res = _brier.rls_score(obs, fct, backend=backend)
    expected = 1.064002
    assert np.isclose(res, expected)
