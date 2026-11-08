import numpy as np
import pytest

import scoringrules as sr

from ._parity import MV_CASES, mv_ensemble_inputs, mv_v_func
from .conftest import assert_inferred


ENSEMBLE_SIZE = 51
N = 100
N_VARS = 30

ESTIMATORS = ["nrg", "fair"]


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_variogram_score(estimator, backend, to_backend, backend_kwargs):
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)

    res = sr.vs_ensemble(
        to_backend(obs), to_backend(fct), estimator=estimator, **backend_kwargs
    )
    assert_inferred(res, backend)
    assert res.shape == (N,)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_variogram_score_permuted_dims(estimator, backend, to_backend, backend_kwargs):
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)

    res = sr.vs_ensemble(
        to_backend(obs), to_backend(fct), estimator=estimator, **backend_kwargs
    )
    # fct (D, N, M) with m_axis=-1, v_axis=0; obs (D, N)
    res_t = sr.vs_ensemble(
        to_backend(obs.T.copy()),
        to_backend(np.moveaxis(fct, (0, 1, 2), (1, 2, 0)).copy()),
        v_axis=0,
        m_axis=-1,
        estimator=estimator,
        **backend_kwargs,
    )
    assert_inferred(res_t, backend)
    np.testing.assert_allclose(np.asarray(res_t), np.asarray(res), rtol=1e-6)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_variogram_score_correctness(estimator, backend_kwargs, native):
    vs = native(sr.vs_ensemble)
    # correctness
    fct = np.array(
        [
            [0.79546742, 0.4777960, 0.2164079, 0.5409873],
            [0.02461368, 0.7584595, 0.3181810, 0.1319422],
        ]
    ).T
    obs = np.array([0.2743836, 0.8146400])

    res = vs(obs, fct, p=0.5, estimator=estimator, **backend_kwargs)
    if estimator == "nrg":
        expected = 0.04114727
        np.testing.assert_allclose(res, expected, rtol=1e-5)
    elif estimator == "fair":
        expected = 0.01407421
        np.testing.assert_allclose(res, expected, rtol=1e-5)

    res = vs(obs, fct, p=1.0, estimator=estimator, **backend_kwargs)
    if estimator == "nrg":
        expected = 0.04480374
        np.testing.assert_allclose(res, expected, rtol=1e-5)
    elif estimator == "fair":
        expected = 0.004730382
        np.testing.assert_allclose(res, expected, rtol=1e-5)

    # with and without dimension weights
    w = np.ones((2, 2))
    res_w = vs(obs, fct, w=w, p=1.0, estimator=estimator, **backend_kwargs)
    np.testing.assert_allclose(res_w, res, rtol=1e-5)

    # with dimension weights
    w = np.array([[0.1, 0.2], [0.2, 0.4]])
    res = vs(obs, fct, w=w, p=0.5, estimator=estimator, **backend_kwargs)
    if estimator == "nrg":
        expected = 0.008229455
        np.testing.assert_allclose(res, expected, atol=1e-6)

    # invariance to diagonal terms of weight matrix
    w = np.array([[1.1, 0.2], [0.2, 100]])
    res = vs(obs, fct, w=w, p=0.5, estimator=estimator, **backend_kwargs)
    if estimator == "nrg":
        np.testing.assert_allclose(res, expected, atol=1e-6)

    # with ensemble weights
    ens_w = np.array([0.1, 0.8, 0.3, 0.4])
    res = vs(obs, fct, ens_w=ens_w, p=0.5, estimator="nrg", **backend_kwargs)
    if estimator == "nrg":
        expected = 0.07648068
        np.testing.assert_allclose(res, expected, atol=1e-6)

    ens_w = np.array([0.9, 0.2, 0.5, 0.5])
    res = vs(obs, fct, ens_w=ens_w, p=0.75, estimator=estimator, **backend_kwargs)
    if estimator == "nrg":
        expected = 0.01160571
        np.testing.assert_allclose(res, expected, atol=1e-6)

    # check invariance to scaling of the weights
    ens_w_scaled = ens_w * 10
    res_scaled = vs(
        obs, fct, ens_w=ens_w_scaled, p=0.75, estimator=estimator, **backend_kwargs
    )
    assert np.allclose(res_scaled, res)

    # with dimension and ensemble weights
    w = np.array([[0.1, 0.2], [0.2, 0.4]])
    ens_w = np.array([0.1, 0.8, 0.3, 0.4])
    res = vs(obs, fct, w=w, ens_w=ens_w, p=0.5, estimator=estimator, **backend_kwargs)
    if estimator == "nrg":
        expected = 0.01529614
        np.testing.assert_allclose(res, expected, atol=1e-6)


@pytest.mark.parametrize("case", MV_CASES)
@pytest.mark.parametrize("M", [6, 7])
@pytest.mark.parametrize("weighted", [False, True], ids=["uw", "w"])
@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("score", ["vs_ensemble", "twvs_ensemble"])
def test_variogram_paths_agree(score, estimator, weighted, M, case, paths_agree):
    obs, fct, kwargs = mv_ensemble_inputs(case, M, weighted)
    args = (mv_v_func,) if score == "twvs_ensemble" else ()
    paths_agree(getattr(sr, score), obs, fct, *args, estimator=estimator, **kwargs)


# p = 1 and 2 take the abs / square branch of the gufuncs, 1.5 the pow branch
@pytest.mark.parametrize("p", [1.0, 2.0, 1.5])
@pytest.mark.parametrize("nan", [False, True], ids=["finite", "nan"])
@pytest.mark.parametrize("weighted", [False, True], ids=["uw", "w"])
@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_variogram_paths_agree_w_p(nan, p, estimator, weighted, paths_agree):
    obs, fct, kwargs = mv_ensemble_inputs("default", 7, weighted)
    if nan:  # a NaN member element in sample 1, a NaN obs element in sample 3
        fct[1, 2, 0] = np.nan
        obs[3, 1] = np.nan
    w = np.random.default_rng(1).uniform(0.1, 2.0, size=(*obs.shape, obs.shape[-1]))
    paths_agree(sr.vs_ensemble, obs, fct, w, p=p, estimator=estimator, **kwargs)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_variogram_ens_w_silent_layout(estimator, native, backend_kwargs):
    """fct (D, M, N) with M == N, m_axis=1, v_axis=0: ens_w (M, N) used to be
    moved along the wrong axis without an error, so the score used the wrong
    weights. Compare with the default layout on moveaxis'd inputs."""
    rng = np.random.default_rng(5)
    D, M = 3, 5
    obs = rng.normal(size=(M, D))  # (N, D) with N == M
    fct = rng.normal(size=(M, M, D))  # (N, M, D)
    ens_w = rng.uniform(0.1, 2.0, size=(M, M))  # (N, M)
    fn = native(sr.vs_ensemble)
    expected = fn(obs, fct, ens_w=ens_w, estimator=estimator, **backend_kwargs)
    res = fn(
        obs.T.copy(),
        np.moveaxis(fct, (0, 1, 2), (2, 1, 0)).copy(),
        m_axis=1,
        v_axis=0,
        ens_w=ens_w.T.copy(),
        estimator=estimator,
        **backend_kwargs,
    )
    np.testing.assert_allclose(np.asarray(res), np.asarray(expected), rtol=1e-6)


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("score", ["vs_ensemble", "twvs_ensemble"])
def test_variogram_ens_w_v_axis_first(score, estimator, native, backend_kwargs):
    """ens_w follows fct without its variables axis when v_axis comes first."""
    obs, fct, kwargs = mv_ensemble_inputs("default", 7, weighted=True)
    args = (lambda x: x,) if score == "twvs_ensemble" else ()
    fn = native(getattr(sr, score))
    expected = fn(obs, fct, *args, estimator=estimator, **kwargs, **backend_kwargs)
    res = fn(
        obs,
        np.swapaxes(fct, -1, -2).copy(),
        *args,
        m_axis=-1,
        v_axis=-2,
        estimator=estimator,
        **kwargs,
        **backend_kwargs,
    )
    np.testing.assert_allclose(np.asarray(res), np.asarray(expected), rtol=1e-6)


def test_variogram_legacy_backend_string_deprecated():
    obs = np.random.randn(4, 3)
    fct = np.random.randn(4, 5, 3)
    w_func = lambda x: x.mean() * 0.0 + 1.0  # noqa: E731
    with pytest.warns(DeprecationWarning):
        sr.vs_ensemble(obs, fct, backend="numpy")
    with pytest.warns(DeprecationWarning) as record:
        sr.twvs_ensemble(obs, fct, lambda x: x, backend="numpy")
    # one warning, pointing at the caller (this file), not at the wrapper
    assert len(record) == 1 and record[0].filename == __file__
    with pytest.warns(DeprecationWarning):
        sr.owvs_ensemble(obs, fct, w_func, backend="numpy")
    with pytest.warns(DeprecationWarning):
        sr.vrvs_ensemble(obs, fct, w_func, backend="numpy")


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_variogram_float32(estimator, backend, to_backend_float32, backend_kwargs):
    rng = np.random.default_rng(3)
    obs = to_backend_float32(rng.normal(size=(6, 4)))
    fct = to_backend_float32(rng.normal(size=(6, 7, 4)))
    ens_w = to_backend_float32(rng.uniform(0.1, 1.0, (6, 7)))
    kwargs = {"estimator": estimator, **backend_kwargs}
    for res in (
        sr.vs_ensemble(obs, fct, **kwargs),
        sr.vs_ensemble(obs, fct, ens_w=ens_w, **kwargs),
        sr.twvs_ensemble(obs, fct, lambda x: x.clip(min=-0.5), **kwargs),
    ):
        assert_inferred(res, backend)
        assert res.dtype == fct.dtype


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_variogram_grad_torch(estimator):
    """p=1: with the default p=0.5 the (zero) diagonal differences |x_i - x_i|**p
    give an infinite derivative, so the gradient is NaN (see PR 5 notes)."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(4)
    obs = torch.as_tensor(rng.normal(size=(4, 3)))
    fct = torch.tensor(rng.normal(size=(4, 7, 3)), requires_grad=True)
    ens_w = torch.as_tensor(rng.uniform(0.1, 1.0, (4, 7)))
    for kwargs in ({}, {"ens_w": ens_w}):
        fct.grad = None
        sr.vs_ensemble(obs, fct, p=1.0, estimator=estimator, **kwargs).sum().backward()
        assert fct.grad is not None and torch.isfinite(fct.grad).all()


def test_mv_numpy_weights_follow_torch_inputs():
    """numpy ``ens_w`` / ``w`` with torch forecasts are converted to the forecast's
    framework, device and float dtype (they used to raise on the array path)."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(7)
    obs, fct = rng.normal(size=(4, 3)), rng.normal(size=(4, 6, 3))
    ens_w, w = rng.uniform(0.1, 1.0, (4, 6)), rng.uniform(0.1, 1.0, (4, 3, 3))
    for dtype in (torch.float32, torch.float64):
        o, f = torch.as_tensor(obs, dtype=dtype), torch.as_tensor(fct, dtype=dtype)
        for res, ref in (
            (sr.es_ensemble(o, f, ens_w=ens_w), sr.es_ensemble(obs, fct, ens_w=ens_w)),
            (
                sr.vs_ensemble(o, f, w, ens_w=ens_w),
                sr.vs_ensemble(obs, fct, w, ens_w=ens_w),
            ),
            (
                sr.owvs_ensemble(o, f, lambda x: x.mean() + 5.0, w, ens_w=ens_w),
                sr.owvs_ensemble(obs, fct, lambda x: x.mean() + 5.0, w, ens_w=ens_w),
            ),
        ):
            assert isinstance(res, torch.Tensor) and res.dtype == dtype
            np.testing.assert_allclose(res.numpy(), ref, rtol=1e-5)


@pytest.mark.parametrize("adversarial", ["outlier0", "offset", "near_identical"])
@pytest.mark.parametrize("weighted", [False, True], ids=["uw", "w"])
@pytest.mark.parametrize(
    "score", ["vs_ensemble", "owvs_ensemble", "vrvs_ensemble"], ids=["vs", "ow", "vr"]
)
def test_variogram_spread_identity_paths_agree(
    score, weighted, adversarial, paths_agree
):
    # the gufuncs sum the member pairs through S0 S2 - S1^2 with shifted sums;
    # inputs where an unshifted identity would cancel badly
    obs, fct, kwargs = mv_ensemble_inputs("default", 9, weighted, D=4)

    def w_func(x):
        return 1.0 / (1.0 + np.exp(-np.mean(x) * 1e-4))

    if adversarial == "outlier0":  # the variogram of member 0 far from the others
        fct[:, 0] += 1e3 * np.arange(4)

        def w_func(x):  # noqa: F811, member 0 gets weight 0
            return float(x[-1] - x[0] < 100.0)

    elif adversarial == "offset":  # |x_i - x_j| large, with a small spread
        obs, fct = obs + 1e4 * np.arange(4), fct + 1e4 * np.arange(4)
    else:
        fct = fct[:, :1] + 1e-6 * fct
    if score == "vs_ensemble":
        paths_agree(sr.vs_ensemble, obs, fct, estimator="fair", rtol=1e-9, **kwargs)
    else:
        paths_agree(getattr(sr, score), obs, fct, w_func, rtol=1e-9, **kwargs)
