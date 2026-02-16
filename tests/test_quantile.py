import numpy as np
import pytest

import scoringrules as sr


def test_quantile_score(backend):
    obs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    fct = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    alpha = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    res = sr.quantile_score(obs, fct, alpha, backend=backend)
    res.__class__.__module__ == backend

    # correctness
    np.isclose(sr.quantile_score(0.3, 0.5, 0.2, backend=backend), 0.16)
    np.isclose(sr.quantile_score(0.3, 0.1, 0.2, backend=backend), 0.04)

    # value checks
    with pytest.raises(ValueError):
        sr.quantile_score(0.3, 0.5, -0.2, backend=backend)
