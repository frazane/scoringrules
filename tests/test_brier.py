import numpy as np
import pytest

import scoringrules as sr

from .conftest import BACKENDS


@pytest.mark.parametrize("backend", BACKENDS)
def test_brier(backend):
    # test exceptions
    with pytest.raises(ValueError):
        sr.brier_score(1, 1.1, backend=backend)

    with pytest.raises(ValueError):
        sr.brier_score(1, -0.1, backend=backend)

    with pytest.raises(ValueError):
        sr.brier_score(0.5, 0.5, backend=backend)

    # test correctness
    obs, fct = 0, 0.271
    res = sr.brier_score(obs, fct, backend=backend)
    expected = 0.073441
    assert np.isclose(res, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_rps(backend):
    # test exceptions
    with pytest.raises(ValueError):
        sr.rps_score(1, 1.1, backend=backend)

    with pytest.raises(ValueError):
        sr.rps_score(1, -0.1, backend=backend)

    # test correctness
    obs, fct = 3, [0.01, 0.32, 0.44, 0.23]
    res = sr.rps_score(obs, fct, backend=backend)
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
    res1 = sr.rps_score(obs, fct, backend=backend)
    fct = np.transpose(fct)
    res2 = sr.rps_score(obs, fct, k_axis=0, backend=backend)
    assert np.allclose(res1, res2)


@pytest.mark.parametrize("backend", BACKENDS)
def test_logs(backend):
    # test exceptions
    with pytest.raises(ValueError):
        sr.log_score(1, 1.1, backend=backend)

    with pytest.raises(ValueError):
        sr.log_score(1, -0.1, backend=backend)

    with pytest.raises(ValueError):
        sr.log_score(0.5, 0.5, backend=backend)

    # test correctness
    obs, fct = 0, 0.481
    res = sr.log_score(obs, fct, backend=backend)
    expected = 0.6558514
    assert np.isclose(res, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_rls(backend):
    # test exceptions
    with pytest.raises(ValueError):
        sr.rls_score(1, 1.1, backend=backend)

    # test exceptions
    with pytest.raises(ValueError):
        sr.rls_score(1, -0.1, backend=backend)

    # test correctness
    obs, fct = 2, [0.21, 0.31, 0.32, 0.16]
    res = sr.rls_score(obs, fct, backend=backend)
    expected = 1.064002
    assert np.isclose(res, expected)

    # test correctness
    obs = [1, 3, 4, 2]
    fct = [
        [0.01, 0.32, 0.44, 0.23],
        [0.12, 0.05, 0.48, 0.35],
        [0.09, 0.21, 0.05, 0.65],
        [0.57, 0.31, 0.08, 0.04],
    ]
    res1 = sr.rls_score(obs, fct, backend=backend)

    fct = np.transpose(fct)
    res2 = sr.rls_score(obs, fct, k_axis=0, backend=backend)

    assert np.allclose(res1, res2)
