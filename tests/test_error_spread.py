import numpy as np
import pytest
from scoringrules._error_spread import error_spread_score

from .conftest import BACKENDS

ENSEMBLE_SIZE = 11
N = 5


@pytest.mark.parametrize("backend", BACKENDS)
def test_error_spread_score(backend):
    obs = np.random.randn(
        N,
    )
    fcts = obs[:, None] + np.random.randn(N, ENSEMBLE_SIZE)
    res = error_spread_score(fcts, obs, backend=backend)

    # approx zero when perfect forecast
    perfect_fcts = obs[..., None] + np.random.randn(N, ENSEMBLE_SIZE) * 1e-5
    res = error_spread_score(perfect_fcts, obs, backend=backend)
    res = np.asarray(res)
    assert not np.any(res - 0.0 > 0.0001)
