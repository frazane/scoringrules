import numpy as np
import pytest
from scoringrules import _crps

ENSEMBLE_SIZE = 51
N = 100

ESTIMATORS = ["nrg", "fair", "pwm", "int", "qd", "akr", "akr_circperm"]
BACKENDS = ["numpy", "numba", "jax"]


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_ensemble(estimator, backend):
    observation = np.random.randn(N)
    mu = observation + np.random.randn(N) * 0.1
    sigma = abs(np.random.randn(N)) * 0.3
    forecasts = np.random.randn(N, ENSEMBLE_SIZE) * sigma[..., None] + mu[..., None]

    # test exceptions
    if backend in ["numpy", "jax"]:
        if estimator not in ["nrg", "fair", "pwm"]:
            with pytest.raises(ValueError):
                _crps.ensemble(
                    forecasts, observation, estimator=estimator, backend=backend
                )
            return

    # non-negative values
    res = _crps.ensemble(forecasts, observation, estimator=estimator, backend=backend)
    assert not np.any(res < 0.0)

    # approx zero when perfect forecast
    perfect_forecasts = (
        observation[..., None] + np.random.randn(N, ENSEMBLE_SIZE) * 0.00001
    )
    res = _crps.ensemble(
        perfect_forecasts, observation, estimator=estimator, backend=backend
    )
    assert not np.any(res - 0.0 > 0.0001)
