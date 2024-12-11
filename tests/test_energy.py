import numpy as np
import pytest
import scoringrules as sr

from .conftest import BACKENDS

ENSEMBLE_SIZE = 11
N = 20
N_VARS = 3


@pytest.mark.parametrize("backend", BACKENDS)
def test_energy_score(backend):
    # test exceptions TODO: add more checks in the code
    with pytest.raises(ValueError):
        obs = np.random.randn(N, N_VARS)
        fct = np.random.randn(N, ENSEMBLE_SIZE, N_VARS - 1)
        sr.energy_score(obs, fct, backend=backend)

    # test shapes
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)
    res = sr.energy_score(obs, fct, backend=backend)
    assert res.shape == (N,)

    # test correctness
    fct = np.array(
        [[0.79546742, 0.4777960, 0.2164079], [0.02461368, 0.7584595, 0.3181810]]
    ).T
    obs = np.array([0.2743836, 0.8146400])
    res = sr.energy_score(obs, fct, backend=backend)
    expected = 0.334542  # TODO: test this against scoringRules
    np.testing.assert_allclose(res, expected, atol=1e-6)
