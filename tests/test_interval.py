import numpy as np
import pytest
import scipy.stats as st
import scoringrules as sr

from .conftest import BACKENDS

N = 100


## We use Bracher et al (2021) Eq. (3) to test the WIS
@pytest.mark.parametrize("backend", BACKENDS)
def test_weighted_interval_score(backend):
    obs = np.zeros(N)
    alpha = np.linspace(0.01, 0.99, 99)
    upper = st.norm(0, 1).ppf(np.tile(1 - alpha / 2, (N, 1)))
    lower = st.norm(0, 1).ppf(np.tile(alpha / 2, (N, 1)))

    WIS = sr.weighted_interval_score(obs, obs, lower, upper, alpha, backend=backend)
    CRPS = sr.crps_normal(obs, 0, 1, backend=backend)
    WIS, CRPS = map(np.asarray, (WIS, CRPS))

    assert np.all(1 - WIS / CRPS <= 0.001 * CRPS)
