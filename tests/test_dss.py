import numpy as np
import pytest
import scoringrules as sr
from ._parity import MV_CASES, UV_CASES, mv_ensemble_inputs, uv_ensemble_inputs
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


# dss has no nan_policy: NaN inputs propagate, so of the nan cases only the
# data of "nan_propagate" is used (without the argument). Its NaN members sit in
# columns 1 and 3, so it needs M >= 4.
DSS_UV_CASES = [c for c in UV_CASES if not c.startswith("nan_")] + ["nan_propagate"]


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("M", [2, 3, 6, 7])
@pytest.mark.parametrize("case", DSS_UV_CASES)
def test_dssuv_paths_agree(case, M, bias, paths_agree):
    if case == "nan_propagate" and M < 4:
        pytest.skip("the nan case puts NaN members in columns 1 and 3, needs M >= 4")
    obs, fct, kwargs = uv_ensemble_inputs(case, M, weighted=False)
    kwargs.pop("nan_policy", None)
    paths_agree(sr.dssuv_ensemble, obs, fct, bias=bias, **kwargs)


# The shared MV cases plus "v_first": v_axis=0 and m_axis=-1, so obs has its
# variables axis moved too (the shared "v_axis" case leaves obs in place).
DSS_MV_CASES = (*MV_CASES, "v_first")


def dss_mv_inputs(case, M, D):
    if case != "v_first":
        return mv_ensemble_inputs(case, M, weighted=False, D=D)
    obs, fct, _ = mv_ensemble_inputs("default", M, weighted=False, D=D)
    obs = np.moveaxis(obs, -1, 0).copy()  # (D, N)
    fct = np.moveaxis(fct, (-1, -2), (0, -1)).copy()  # (D, N, M)
    return obs, fct, {"m_axis": -1, "v_axis": 0}


@pytest.mark.parametrize("bias", [False, True])
# the covariance is singular unless M > D, so tiny ensembles get fewer variables
@pytest.mark.parametrize("M,D", [(2, 1), (3, 2), (6, 1), (6, 3), (7, 3)])
@pytest.mark.parametrize("case", DSS_MV_CASES)
def test_dssmv_paths_agree(case, M, D, bias, paths_agree):
    obs, fct, kwargs = dss_mv_inputs(case, M, D)
    if case == "int_fct" and (M, D) == (3, 2):
        # some integer ensembles of 3 members in 2-d are collinear, so the
        # covariance is exactly singular and both numpy paths raise
        kwargs["raises"] = np.linalg.LinAlgError
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
