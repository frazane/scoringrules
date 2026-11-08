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
    # some integer ensembles of 3 members in 2-d are collinear: their covariance is
    # singular and both paths give NaN there
    paths_agree(sr.dssmv_ensemble, obs, fct, bias=bias, **kwargs)


def _dssmv_degenerate_inputs():
    """(obs, fct) of 7 samples, M=8, D=3: sample 0 is regular, 1 has a NaN member
    element, 2 an inf member element, 3 a duplicated variable (collinear
    members), 4 a variable that is a linear combination of the others, 5 a NaN
    obs, 6 an inf obs. Each gives NaN except sample 0."""
    rng = np.random.default_rng(7)
    obs = rng.normal(size=(7, 3))
    fct = rng.normal(size=(7, 8, 3))
    fct[1, 2, 0] = np.nan
    fct[2, 5, 1] = np.inf
    fct[3, :, 2] = fct[3, :, 0]
    fct[4, :, 2] = 0.3 * fct[4, :, 0] - 1.7 * fct[4, :, 1]
    obs[5, 1] = np.nan
    obs[6, 0] = np.inf
    return obs, fct


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
@pytest.mark.parametrize("bias", [False, True])
def test_dssmv_degenerate_covariance_is_nan(bias, backend, native):
    # a NaN or inf member, or a singular covariance gives NaN (not an error) on
    # every path; the regular sample is unaffected
    obs, fct = _dssmv_degenerate_inputs()
    res = np.asarray(native(sr.dssmv_ensemble)(obs, fct, bias=bias))
    expected = [
        d @ np.linalg.inv(c) @ d + np.log(np.linalg.det(c))
        for d, c in [
            (obs[0] - fct[0].mean(axis=0), np.cov(fct[0], rowvar=False, bias=bias))
        ]
    ]
    rtol = 1e-4 if backend == "jax" else 1e-10
    np.testing.assert_allclose(res[:1], expected, rtol=rtol)
    assert np.isnan(res[1:]).all(), res


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
@pytest.mark.parametrize("M,D", [(1, 1), (1, 3), (2, 2), (3, 4)])
@pytest.mark.parametrize("bias", [False, True])
def test_dssmv_too_few_members_is_nan(M, D, bias, backend, native):
    # M <= D: the centred members span at most M - 1 dimensions
    rng = np.random.default_rng(8)
    res = native(sr.dssmv_ensemble)(
        rng.normal(size=(4, D)), rng.normal(size=(4, M, D)), bias=bias
    )
    assert res.shape == (4,)
    assert np.isnan(np.asarray(res)).all()


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
@pytest.mark.parametrize("bias", [False, True])
def test_dssmv_degenerate_paths_agree(bias, paths_agree):
    obs, fct = _dssmv_degenerate_inputs()
    paths_agree(sr.dssmv_ensemble, obs, fct, bias=bias)
    rng = np.random.default_rng(9)
    obs, fct = rng.normal(size=(3, 4)), rng.normal(size=(3, 4, 4))  # M == D
    paths_agree(sr.dssmv_ensemble, obs, fct, bias=bias, allow_all_nan=True)


def test_dssmv_tol_factor_matches():
    # the gufunc keeps its own copy of the constants (numba's cache tracks its file)
    pytest.importorskip("numba")
    from scoringrules.core.dss import _gufuncs, _score

    assert _gufuncs.TOL_FACTOR == _score.TOL_FACTOR
    assert _gufuncs.ZERO_VAR_FACTOR == _score.ZERO_VAR_FACTOR


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
@pytest.mark.parametrize("value", [0.0, 0.1, 2.5, 273.15, 1e-3, 101325.0])
@pytest.mark.parametrize("M", [10, 21, 60])
def test_dssmv_constant_variable_is_nan(value, M, backend, native):
    # a constant variable whose mean does not round exactly to its value has a
    # rounding-level variance; it must still count as constant on every path
    rng = np.random.default_rng(11)
    fct = rng.normal(size=(3, M, 4))
    fct[:, :, 2] = value
    obs = rng.normal(size=(3, 4))
    res = np.asarray(native(sr.dssmv_ensemble)(obs, fct))
    assert np.isnan(res).all(), res
    if backend in ("numpy", "numba"):
        f32 = np.asarray(
            native(sr.dssmv_ensemble)(obs.astype(np.float32), fct.astype(np.float32))
        )
        assert np.isnan(f32).all(), f32


@pytest.mark.parametrize("center,spread", [(1e5, 1e-3), (273.15, 1e-2), (1.0, 1e-6)])
def test_dssmv_small_spread_is_not_constant(center, spread, paths_agree):
    # the zero-variance floor is relative to eps: a 1e5 pressure with a 1e-3 spread
    # is a regular float64 variable
    rng = np.random.default_rng(12)
    fct = rng.normal(size=(20, 21, 3))
    fct[..., 2] = center + spread * fct[..., 2]
    obs = fct[:, 0] + 0.1 * spread
    res, _ = paths_agree(sr.dssmv_ensemble, obs, fct, rtol=1e-8)
    assert np.isfinite(res).all()


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_dssmv_float32_judged_at_float32_precision(paths_agree):
    # a variable that is a linear combination of the others up to a 1e-4 relative
    # perturbation: singular at float32 precision (NaN on both paths), regular in
    # float64
    rng = np.random.default_rng(13)
    fct = rng.normal(size=(20, 21, 3))
    fct[..., 2] = fct[..., 0] + fct[..., 1] + 1e-4 * rng.normal(size=(20, 21))
    obs = rng.normal(size=(20, 3))
    r32 = sr.dssmv_ensemble(
        obs.astype(np.float32), fct.astype(np.float32), backend="numba"
    )
    paths_agree(
        sr.dssmv_ensemble,
        obs.astype(np.float32),
        fct.astype(np.float32),
        allow_all_nan=True,
    )
    assert np.isnan(r32).all()
    # ill-conditioned in float64 (pivot ratio ~5e-9), so the paths differ ~1e-7
    res, _ = paths_agree(sr.dssmv_ensemble, obs, fct, rtol=1e-5)
    assert np.isfinite(res).all()


def test_dssmv_float32_gufunc_matches_float64(requires_numba):
    # the gufunc accumulates in float64 for float32 input and returns float32
    rng = np.random.default_rng(10)
    obs, fct = rng.normal(size=(20, 3)), rng.normal(size=(20, 11, 3))
    r32 = sr.dssmv_ensemble(
        obs.astype(np.float32), fct.astype(np.float32), backend="numba"
    )
    r64 = sr.dssmv_ensemble(
        obs.astype(np.float32).astype(np.float64),
        fct.astype(np.float32).astype(np.float64),
        backend="numba",
    )
    assert r32.dtype == np.float32
    np.testing.assert_allclose(r32, r64.astype(np.float32), rtol=1e-6)


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
