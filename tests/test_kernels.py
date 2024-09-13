import numpy as np
import pytest
from scoringrules import _kernels

from .conftest import BACKENDS

ENSEMBLE_SIZE = 51
N = 100

ESTIMATORS = ["nrg", "fair"]


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_gks(estimator, backend):
    obs = np.random.randn(N)
    mu = obs + np.random.randn(N) * 0.1
    sigma = abs(np.random.randn(N)) * 0.3
    fct = np.random.randn(N, ENSEMBLE_SIZE) * sigma[..., None] + mu[..., None]

    # non-negative values
    res = _kernels.gks_ensemble(obs, fct, estimator=estimator, backend=backend)
    res = np.asarray(res)
    assert not np.any(res < 0.0)

    if estimator == "nrg":
        # approx zero when perfect forecast
        perfect_fct = obs[..., None] + np.random.randn(N, ENSEMBLE_SIZE) * 0.00001
        res = _kernels.gks_ensemble(
            obs, perfect_fct, estimator=estimator, backend=backend
        )
        res = np.asarray(res)
        assert not np.any(res - 0.0 > 0.0001)

        # test correctness
        obs, fct = 11.6, np.array([9.8, 8.7, 11.9, 12.1, 13.4])
        res = _kernels.gks_ensemble(obs, fct, estimator=estimator, backend=backend)
        expected = 0.2490516
        assert np.isclose(res, expected)

    elif estimator == "fair":
        # test correctness
        obs, fct = 11.6, np.array([9.8, 8.7, 11.9, 12.1, 13.4])
        res = _kernels.gks_ensemble(obs, fct, estimator=estimator, backend=backend)
        expected = 0.2987752
        assert np.isclose(res, expected)
