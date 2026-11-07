import numpy as np
import pytest
import scoringrules as sr
from ._parity import UV_CASES, uv_ensemble_inputs
from .conftest import assert_inferred


ENSEMBLE_SIZE = 11
N = 20
N_VARS = 3


def assert_close(result, expected, backend, *, rtol=1e-6, atol=1e-6):
    """Compare a framework-native result to a numpy/scalar value; jax runs in
    float32 in the test environment, so its tolerances are loosened."""
    if backend == "jax":
        rtol, atol = max(rtol, 1e-4), max(atol, 1e-5)
    np.testing.assert_allclose(np.asarray(result), expected, rtol=rtol, atol=atol)


def test_ds_score_uv(backend, to_backend, backend_kwargs):
    # m_axis keyword
    res = sr.dssuv_ensemble(
        to_backend(np.random.randn(N)),
        to_backend(np.random.randn(ENSEMBLE_SIZE, N)),
        m_axis=0,
        **backend_kwargs,
    )
    assert_inferred(res, backend)
    assert res.shape == (N,)

    # test correctness
    obs, fct = 11.6, np.array([9.8, 8.7, 11.9, 12.1, 13.4])
    res = sr.dssuv_ensemble(obs, to_backend(fct), bias=True, **backend_kwargs)
    assert_close(res, 1.115645, backend)
    res = sr.dssuv_ensemble(obs, to_backend(fct), **backend_kwargs)
    sig = np.std(fct, ddof=1)
    assert_close(res, ((obs - fct.mean()) / sig) ** 2 + 2 * np.log(sig), backend)


def test_ds_score_mv(backend, to_backend, backend_kwargs):
    with pytest.raises(ValueError):
        obs = np.random.randn(N, N_VARS)
        fct = np.random.randn(N, ENSEMBLE_SIZE, N_VARS - 1)
        sr.dssmv_ensemble(to_backend(obs), to_backend(fct), **backend_kwargs)

    # test shapes
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)
    res = sr.dssmv_ensemble(to_backend(obs), to_backend(fct), **backend_kwargs)
    assert_inferred(res, backend)
    assert res.shape == (N,)

    # test correctness
    fct = np.array(
        [[0.79546742, 0.4777960, 0.2164079], [0.02461368, 0.7584595, 0.3181810]]
    ).T
    obs = np.array([0.2743836, 0.8146400])
    res = sr.dssmv_ensemble(to_backend(obs), to_backend(fct), **backend_kwargs)
    assert_close(res, -3.161553, backend, rtol=0)

    # batched result matches per-element calls
    fct = np.random.randn(2, 3, 100, 4)
    obs = np.random.randn(2, 3, 4)
    res = sr.dssmv_ensemble(to_backend(obs), to_backend(fct), **backend_kwargs)
    assert_inferred(res, backend)
    assert res.shape == (2, 3)
    for i in range(2):
        for j in range(3):
            res_ij = sr.dssmv_ensemble(
                to_backend(obs[i, j, :]), to_backend(fct[i, j, :, :]), **backend_kwargs
            )
            assert_close(res[i, j], res_ij, backend)


def test_ds_score_mv_bias(backend, to_backend, backend_kwargs):
    rng = np.random.default_rng(1)
    obs = rng.normal(size=(4, N_VARS))
    fct = rng.normal(size=(4, ENSEMBLE_SIZE, N_VARS))
    for bias in (False, True):
        res = sr.dssmv_ensemble(
            to_backend(obs), to_backend(fct), bias=bias, **backend_kwargs
        )
        assert_inferred(res, backend)
        expected = []
        for o, f in zip(obs, fct):
            cov = np.cov(f, rowvar=False, bias=bias)
            d = o - f.mean(axis=0)
            expected.append(d @ np.linalg.inv(cov) @ d + np.log(np.linalg.det(cov)))
        assert_close(res, expected, backend)


def test_dss_legacy_backend_string_deprecated():
    obs = np.random.randn(N, N_VARS)
    fct = np.random.randn(N, ENSEMBLE_SIZE, N_VARS)
    with pytest.warns(DeprecationWarning):
        sr.dssuv_ensemble(obs[:, 0], fct[..., 0], backend="numpy")
    with pytest.warns(DeprecationWarning):
        sr.dssmv_ensemble(obs, fct, backend="numpy")


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("M", [6, 7])
@pytest.mark.parametrize("case", UV_CASES)
def test_dssuv_paths_agree(case, M, bias, paths_agree):
    obs, fct, kwargs = uv_ensemble_inputs(case, M, weighted=False)
    paths_agree(sr.dssuv_ensemble, obs, fct, bias=bias, **kwargs)


MV_CASES = ("default", "d1", "n1", "int_obs", "single", "batch", "m_first", "v_first")


def mv_ensemble_inputs(case, M, D=3, seed=0, N=5):
    """``(obs, fct, kwargs)`` for a multivariate ensemble parity case.

    ``"default"``: ``fct`` of shape ``(N, M, D)``; ``"d1"``: ``D=1``; ``"n1"``:
    ``N=1``; ``"int_obs"``: integer ``obs``; ``"single"``: ``fct`` of shape
    ``(M, D)`` (0-d result); ``"batch"``: ``fct`` of shape ``(2, N, M, D)``;
    ``"m_first"``: ``fct`` of shape ``(M, N, D)`` with ``m_axis=0``;
    ``"v_first"``: ``fct`` of shape ``(D, N, M)`` with ``m_axis=-1, v_axis=0``.
    """
    rng = np.random.default_rng(seed)
    D = 1 if case == "d1" else D
    batch = {"n1": (1,), "single": (), "batch": (2, N)}.get(case, (N,))
    obs = rng.normal(size=(*batch, D))
    fct = rng.normal(size=(*batch, M, D))
    kwargs = {}
    if case == "int_obs":
        obs = rng.integers(-2, 3, size=(*batch, D))
    elif case == "m_first":
        fct = np.moveaxis(fct, -2, 0).copy()
        kwargs = {"m_axis": 0}
    elif case == "v_first":
        obs = np.moveaxis(obs, -1, 0).copy()
        fct = np.moveaxis(fct, (-1, -2), (0, -1)).copy()
        kwargs = {"m_axis": -1, "v_axis": 0}
    return obs, fct, kwargs


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("M", [6, 7])
@pytest.mark.parametrize("case", MV_CASES)
def test_dssmv_paths_agree(case, M, bias, paths_agree):
    obs, fct, kwargs = mv_ensemble_inputs(case, M)
    paths_agree(sr.dssmv_ensemble, obs, fct, bias=bias, **kwargs)


def test_dss_float32(backend, to_backend_float32, backend_kwargs):
    rng = np.random.default_rng(2)
    obs = to_backend_float32(rng.normal(size=(N, N_VARS)))
    fct = to_backend_float32(rng.normal(size=(N, ENSEMBLE_SIZE, N_VARS)))
    for res, ref in (
        (sr.dssuv_ensemble(obs[:, 0], fct[..., 0], **backend_kwargs), fct),
        (sr.dssmv_ensemble(obs, fct, **backend_kwargs), fct),
    ):
        assert_inferred(res, backend)
        assert res.dtype == ref.dtype


def test_dss_grad_torch():
    torch = pytest.importorskip("torch")
    obs = torch.randn(4, N_VARS, dtype=torch.float64)
    fct = torch.randn(4, ENSEMBLE_SIZE, N_VARS, dtype=torch.float64, requires_grad=True)
    sr.dssuv_ensemble(obs[:, 0], fct[..., 0]).sum().backward()
    assert fct.grad is not None and torch.isfinite(fct.grad).all()
    fct.grad = None
    sr.dssmv_ensemble(obs, fct).sum().backward()
    assert fct.grad is not None and torch.isfinite(fct.grad).all()
