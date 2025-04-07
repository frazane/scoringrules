import numpy as np
import pytest
import scoringrules as sr

from .conftest import BACKENDS

ENSEMBLE_SIZE = 11
N = 5


@pytest.mark.parametrize("backend", BACKENDS)
def test_error_spread_score(backend):
    # test shape
    obs = np.random.randn(N)
    fct = obs[:, None] + np.random.randn(N, ENSEMBLE_SIZE)
    res = sr.error_spread_score(obs, fct, backend=backend)
    assert res.shape == (N,)

    obs = np.random.randn(N)
    fct = np.random.randn(ENSEMBLE_SIZE, N)
    res = sr.error_spread_score(obs, fct, m_axis=0, backend=backend)
    assert res.shape == (N,)

    # test correctness
    obs = 0.5
    fct = np.sqrt(np.arange(ENSEMBLE_SIZE))
    res = sr.error_spread_score(obs, fct, backend=backend)
    assert np.isclose(res, 1.7942340)

    # approx zero when perfect forecast
    obs = np.random.randn(N)
    perfect_fct = obs[..., None] + np.random.randn(N, ENSEMBLE_SIZE) * 1e-5
    res = sr.error_spread_score(obs, perfect_fct, backend=backend)
    res = np.asarray(res)
    assert not np.any(res - 0.0 > 0.0001)
