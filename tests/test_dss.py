import numpy as np
import pytest
import scoringrules as sr

from .conftest import BACKENDS

ENSEMBLE_SIZE = 11
N = 20
N_VARS = 3


@pytest.mark.parametrize("backend", BACKENDS)
def test_ds_score_uv(backend):
    obs = np.random.randn(N)
    mu = obs + np.random.randn(N) * 0.1
    sigma = abs(np.random.randn(N)) * 0.3
    fct = np.random.randn(N, ENSEMBLE_SIZE) * sigma[..., None] + mu[..., None]

    # m_axis keyword
    res = sr.dssuv_ensemble(
        obs,
        np.random.randn(ENSEMBLE_SIZE, N),
        m_axis=0,
        backend=backend,
    )
    res = np.asarray(res)
    assert res.shape == (N,)

    # test correctness
    obs, fct = 11.6, np.array([9.8, 8.7, 11.9, 12.1, 13.4])
    res = sr.dssuv_ensemble(obs, fct, bias=True, backend=backend)
    expected = 1.115645
    assert np.isclose(res, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_ds_score_mv(backend):
    with pytest.raises(ValueError):
        obs = np.random.randn(N, N_VARS)
        fct = np.random.randn(N, ENSEMBLE_SIZE, N_VARS - 1)
        sr.dssmv_ensemble(obs, fct, backend=backend)

    # test shapes
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)
    res = sr.dssmv_ensemble(obs, fct, backend=backend)
    assert res.shape == (N,)

    # test correctness
    fct = np.array(
        [[0.79546742, 0.4777960, 0.2164079], [0.02461368, 0.7584595, 0.3181810]]
    ).T
    obs = np.array([0.2743836, 0.8146400])
    res = sr.dssmv_ensemble(obs, fct, backend=backend)
    expected = -3.161553
    np.testing.assert_allclose(res, expected, atol=1e-6)

    # test correctness
    fct = np.random.randn(2, 3, 100, 4)
    obs = np.random.randn(2, 3, 4)
    res = sr.dssmv_ensemble(obs, fct, backend=backend)
    for i in range(2):
        for j in range(3):
            res_ij = sr.dssmv_ensemble(obs[i, j, :], fct[i, j, :, :], backend=backend)
            np.testing.assert_allclose(res[i, j], res_ij, atol=1e-6)
