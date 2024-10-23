import numpy as np
import pytest
import scipy.stats as st
import scoringrules as sr

from .conftest import BACKENDS

N = 100


@pytest.mark.parametrize("backend", BACKENDS)
def test_interval_score(backend):
    # basic functionality
    _ = sr.interval_score(0.1, 0.0, 0.4, 0.5)

    # shapes
    res = sr.interval_score(
        obs=np.array([0.1, 0.2, 0.3]),
        lower=np.array([0.0, 0.1, 0.2]),
        upper=np.array([0.4, 0.3, 0.5]),
        alpha=0.5,
        backend=backend,
    )
    assert res.shape == (3,)

    res = sr.interval_score(
        obs=np.random.uniform(size=(10,)),
        lower=np.ones((10, 5)) * 0.2,
        upper=np.ones((10, 5)) * 0.8,
        alpha=np.linspace(0.1, 0.9, 5),
        backend=backend,
    )
    assert res.shape == (10, 5)

    # raise ValueError
    with pytest.raises(ValueError):
        _ = sr.interval_score(
            obs=np.random.uniform(size=(10,)),
            lower=np.ones((10, 5)) * 0.2,
            upper=np.ones((10, 4)) * 0.8,
            alpha=np.linspace(0.1, 0.9, 5),
            backend=backend,
        )

    with pytest.raises(ValueError):
        _ = sr.interval_score(
            obs=np.random.uniform(size=(10,)),
            lower=np.ones((10, 5)) * 0.2,
            upper=np.ones((10, 5)) * 0.8,
            alpha=np.linspace(0.1, 0.9, 4),
            backend=backend,
        )

    # correctness
    res = sr.interval_score(0.1, 0.0, 0.4, 0.5, backend=backend)
    np.isclose(res, 0.4)


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
