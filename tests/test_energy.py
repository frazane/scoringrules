import numpy as np
import pytest
import scoringrules as sr

from ._parity import MV_CASES, mv_ensemble_inputs, mv_v_func
from .conftest import assert_inferred


ENSEMBLE_SIZE = 11
N = 20
N_VARS = 3

ESTIMATORS = ["nrg", "fair", "akr", "akr_circperm"]


def assert_close(result, expected, backend, *, rtol=1e-7, atol=1e-6):
    """Compare a framework-native result to a numpy/scalar value; jax runs in
    float32 in the test environment, so its tolerances are loosened."""
    if backend == "jax":
        rtol, atol = max(rtol, 1e-4), max(atol, 1e-5)
    np.testing.assert_allclose(np.asarray(result), expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_energy_score(estimator, backend, to_backend, backend_kwargs):
    kwargs = {"estimator": estimator, **backend_kwargs}

    # incorrect dimensions
    with pytest.raises(ValueError):
        obs = np.random.randn(N, N_VARS)
        fct = np.random.randn(N, ENSEMBLE_SIZE, N_VARS - 1)
        sr.es_ensemble(to_backend(obs), to_backend(fct), **kwargs)

    # undefined estimator
    with pytest.raises(ValueError):
        fct = np.random.randn(N, ENSEMBLE_SIZE, N_VARS)
        sr.es_ensemble(
            to_backend(obs),
            to_backend(fct),
            estimator="undefined_estimator",
            **backend_kwargs,
        )

    # shapes
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)
    res = sr.es_ensemble(to_backend(obs), to_backend(fct), **kwargs)
    assert_inferred(res, backend)
    assert res.shape == (N,)

    # equivalence with and without weights
    w = np.ones(fct.shape[:-1])
    res_w = sr.es_ensemble(
        to_backend(obs), to_backend(fct), ens_w=to_backend(w), **kwargs
    )
    assert_inferred(res_w, backend)
    assert_close(res_w, np.asarray(res), backend, rtol=1e-10)

    # correctness
    fct = np.array(
        [
            [0.79546742, 0.4777960, 0.2164079, 0.5409873],
            [0.02461368, 0.7584595, 0.3181810, 0.1319422],
        ]
    ).T
    obs = np.array([0.2743836, 0.8146400])
    expected = {
        "nrg": 0.3949793,
        "fair": 0.3274585,
        "akr": 0.3522818,
        "akr_circperm": 0.2778118,
    }
    res = sr.es_ensemble(to_backend(obs), to_backend(fct), **kwargs)
    assert_close(res, expected[estimator], backend)

    # correctness with weights
    w = np.array([0.1, 0.2, 0.3, 0.4])
    expected = {
        "nrg": 0.4074382,
        "fair": 0.333500969,
        "akr": 0.3345371,
        "akr_circperm": 0.2612048,
    }
    res = sr.es_ensemble(
        to_backend(obs), to_backend(fct), ens_w=to_backend(w), **kwargs
    )
    assert_close(res, expected[estimator], backend)

    # invariance to scaling of the weights
    res_scaled = sr.es_ensemble(
        to_backend(obs), to_backend(fct), ens_w=to_backend(w * 10), **kwargs
    )
    assert_close(res_scaled, np.asarray(res), backend)


OMIT_ESTIMATORS = ["nrg", "fair"]


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_energy_score_nan_policy(estimator, backend_kwargs, native):
    """es_ensemble with NaN forecast members, no user weights."""
    kwargs = {"estimator": estimator, **backend_kwargs}
    es = native(sr.es_ensemble)

    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)

    fct_nan = fct.copy()
    fct_nan[0, [0, 3, 6], 0] = np.nan
    fct_nan[2, [5], 1] = np.nan
    nan_positions = np.isnan(fct_nan).any(axis=(1, 2))

    # default + explicit 'propagate': corrupted samples return NaN
    for policy in (None, "propagate"):
        pol_kwargs = kwargs if policy is None else {**kwargs, "nan_policy": policy}
        res = np.asarray(es(obs, fct_nan, **pol_kwargs))
        assert np.all(np.isnan(res[nan_positions]))
        assert not np.any(np.isnan(res[~nan_positions]))

    # 'raise'
    with pytest.raises(ValueError):
        es(obs, fct_nan, nan_policy="raise", **kwargs)

    # 'omit' unsupported for akr / akr_circperm
    if estimator not in OMIT_ESTIMATORS:
        with pytest.raises(NotImplementedError):
            es(obs, fct_nan, nan_policy="omit", **kwargs)
        return

    # 'omit': no NaN in result
    res = np.asarray(es(obs, fct_nan, nan_policy="omit", **kwargs))
    assert not np.any(np.isnan(res))

    # 'omit': equivalence with the surviving sub-ensemble, per row
    for i in range(fct.shape[0]):
        valid = ~np.isnan(fct_nan[i]).any(axis=-1)
        fct_clean = fct_nan[i][valid]
        res_clean = es(obs[i], fct_clean, **kwargs)
        res_omit = es(obs[i], fct_nan[i], nan_policy="omit", **kwargs)
        assert np.allclose(res_omit, res_clean)

    # all-members-NaN row: 'omit' yields NaN (documented 0/0 edge case)
    fct_allnan = fct.copy()
    fct_allnan[3, :, 0] = np.nan
    res_allnan = np.asarray(es(obs, fct_allnan, nan_policy="omit", **kwargs))
    assert np.isnan(res_allnan[3])

    # NaN in obs always propagates, even under 'omit'
    obs_nan = obs.copy()
    obs_nan[1, 0] = np.nan
    res = np.asarray(es(obs_nan, fct, nan_policy="omit", **kwargs))
    assert np.isnan(res[1])


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_energy_score_w_ens_nan_policy(estimator, backend_kwargs, native):
    """es_ensemble with NaN forecast members AND user-supplied ens_w."""
    kwargs = {"estimator": estimator, **backend_kwargs}
    es = native(sr.es_ensemble)

    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)
    uniform_ens_w = np.ones(fct.shape[:-1])
    non_uniform_ens_w = np.random.rand(*fct.shape[:-1])

    fct_nan = fct.copy()
    fct_nan[0, [0, 3, 6], 0] = np.nan
    fct_nan[2, [5], 1] = np.nan
    nan_positions = np.isnan(fct_nan).any(axis=(1, 2))

    # 'propagate' (default + explicit)
    for policy in (None, "propagate"):
        pol_kwargs = kwargs if policy is None else {**kwargs, "nan_policy": policy}
        res = np.asarray(es(obs, fct_nan, ens_w=uniform_ens_w, **pol_kwargs))
        assert np.all(np.isnan(res[nan_positions]))

    # 'raise'
    with pytest.raises(ValueError):
        es(obs, fct_nan, ens_w=uniform_ens_w, nan_policy="raise", **kwargs)

    # 'omit' unsupported for akr / akr_circperm
    if estimator not in OMIT_ESTIMATORS:
        with pytest.raises(NotImplementedError):
            es(obs, fct_nan, ens_w=uniform_ens_w, nan_policy="omit", **kwargs)
        return

    # 'omit' with uniform weights: no NaN
    res = np.asarray(es(obs, fct_nan, ens_w=uniform_ens_w, nan_policy="omit", **kwargs))
    assert not np.any(np.isnan(res))

    # 'omit' with non-uniform weights: weights actually matter
    res = np.asarray(es(obs, fct, ens_w=non_uniform_ens_w, nan_policy="omit", **kwargs))
    res_nans = np.asarray(
        es(obs, fct_nan, ens_w=non_uniform_ens_w, nan_policy="omit", **kwargs)
    )
    assert not np.any(np.isnan(res_nans))
    assert not np.allclose(res[nan_positions], res_nans[nan_positions])
    assert np.allclose(res[~nan_positions], res_nans[~nan_positions])

    # 'omit': equivalence with clean weighted sub-ensemble, per row
    for i in range(fct.shape[0]):
        valid = ~np.isnan(fct_nan[i]).any(axis=-1)
        res_clean = es(
            obs[i],
            fct_nan[i][valid],
            ens_w=uniform_ens_w[i][valid],
            nan_policy="omit",
            **kwargs,
        )
        res_omit = es(
            obs[i], fct_nan[i], ens_w=uniform_ens_w[i], nan_policy="omit", **kwargs
        )
        assert np.allclose(res_omit, res_clean)

    # 'omit' with non-uniform weights: equivalence with clean weighted sub-ensemble
    for i in range(fct.shape[0]):
        valid = ~np.isnan(fct_nan[i]).any(axis=-1)
        res_clean = es(
            obs[i],
            fct_nan[i][valid],
            ens_w=non_uniform_ens_w[i][valid],
            nan_policy="omit",
            **kwargs,
        )
        res_omit = es(
            obs[i], fct_nan[i], ens_w=non_uniform_ens_w[i], nan_policy="omit", **kwargs
        )
        assert np.allclose(res_omit, res_clean)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_energy_score_nan_weights(estimator, backend_kwargs, native):
    """es_ensemble when ens_w itself contains NaN (forecasts are clean)."""
    kwargs = {"estimator": estimator, **backend_kwargs}
    es = native(sr.es_ensemble)

    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)
    ens_w = np.random.rand(N, ENSEMBLE_SIZE)
    ens_w[0, [0, 3, 6]] = np.nan
    ens_w[2, [5]] = np.nan
    nan_positions = np.isnan(ens_w).any(axis=1)

    # 'propagate'
    res = np.asarray(es(obs, fct, ens_w=ens_w, nan_policy="propagate", **kwargs))
    assert np.all(np.isnan(res[nan_positions]))

    # 'raise'
    with pytest.raises(ValueError):
        es(obs, fct, ens_w=ens_w, nan_policy="raise", **kwargs)

    # 'omit' unsupported for akr / akr_circperm
    if estimator not in OMIT_ESTIMATORS:
        with pytest.raises(NotImplementedError):
            es(obs, fct, ens_w=ens_w, nan_policy="omit", **kwargs)
        return

    # 'omit': NaN-weighted members get zero weight (dropped)
    res = np.asarray(es(obs, fct, ens_w=ens_w, nan_policy="omit", **kwargs))
    assert not np.any(np.isnan(res))
    for i in range(fct.shape[0]):
        valid = ~np.isnan(ens_w[i])
        res_clean = es(
            obs[i], fct[i][valid], ens_w=ens_w[i][valid], nan_policy="omit", **kwargs
        )
        res_omit = es(obs[i], fct[i], ens_w=ens_w[i], nan_policy="omit", **kwargs)
        assert np.allclose(res_omit, res_clean)


@pytest.mark.parametrize("case", MV_CASES)
@pytest.mark.parametrize("M", [6, 7])
@pytest.mark.parametrize("weighted", [False, True], ids=["uw", "w"])
@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("score", ["es_ensemble", "twes_ensemble"])
def test_energy_paths_agree(score, estimator, weighted, M, case, paths_agree):
    obs, fct, kwargs = mv_ensemble_inputs(case, M, weighted)
    args = (mv_v_func,) if score == "twes_ensemble" else ()
    paths_agree(getattr(sr, score), obs, fct, *args, estimator=estimator, **kwargs)


def test_energy_score_akr_circperm_ens_w_known_value(
    backend, to_backend, backend_kwargs
):
    """Weighted akr_circperm pairs member i with (i + M // 2) % M and uses w_i (#149)."""
    obs = np.array([0.3, -0.1])
    fct = np.array([[0.1, 0.2], [-0.4, 0.0], [1.2, -0.5]])
    w = [0.5, 0.3, 0.2]
    partner = [1, 2, 0]  # M = 3, M // 2 = 1

    def norm(a, b):
        return sum((ai - bi) ** 2 for ai, bi in zip(a, b)) ** 0.5

    e_1 = sum(w[i] * norm(fct[i], obs) for i in range(3))
    e_2 = sum(w[i] * norm(fct[i], fct[partner[i]]) for i in range(3))
    expected = e_1 - 0.5 * e_2
    res = sr.es_ensemble(
        to_backend(obs),
        to_backend(fct),
        ens_w=to_backend(np.array(w)),
        estimator="akr_circperm",
        **backend_kwargs,
    )
    assert_close(res, expected, backend)


@pytest.mark.parametrize("weighted", [False, True], ids=["uw", "w"])
@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("score", ["es_ensemble", "twes_ensemble"])
def test_energy_ens_w_v_axis_first(score, estimator, weighted, native, backend_kwargs):
    """ens_w follows fct without its variables axis, also when v_axis comes
    before m_axis (it used to raise: the weight axis was taken against fct.ndim)."""
    obs, fct, kwargs = mv_ensemble_inputs("default", 7, weighted)
    args = (lambda x: x,) if score == "twes_ensemble" else ()
    fn = native(getattr(sr, score))
    expected = fn(obs, fct, *args, estimator=estimator, **kwargs, **backend_kwargs)
    kwargs = {**kwargs, "m_axis": -1, "v_axis": -2}
    fct_t = np.swapaxes(fct, -1, -2).copy()
    res = fn(obs, fct_t, *args, estimator=estimator, **kwargs, **backend_kwargs)
    np.testing.assert_allclose(np.asarray(res), np.asarray(expected), rtol=1e-6)
    # m_axis=0 with v_axis=1: fct (M, D, N), obs (D, N), ens_w (M, N)
    fct_t = np.moveaxis(fct, (-2, -1), (0, 1)).copy()
    kwargs = {**kwargs, "m_axis": 0, "v_axis": 1}
    if weighted:
        kwargs["ens_w"] = np.moveaxis(kwargs["ens_w"], -1, 0).copy()
    res = fn(
        obs.T.copy(), fct_t, *args, estimator=estimator, **kwargs, **backend_kwargs
    )
    np.testing.assert_allclose(np.asarray(res), np.asarray(expected), rtol=1e-6)


def test_energy_legacy_backend_string_deprecated():
    obs = np.random.randn(N, N_VARS)
    fct = np.random.randn(N, ENSEMBLE_SIZE, N_VARS)
    with pytest.warns(DeprecationWarning):
        sr.es_ensemble(obs, fct, backend="numpy")
    with pytest.warns(DeprecationWarning) as record:
        sr.twes_ensemble(obs, fct, lambda x: x, backend="numpy")
    # one warning, pointing at the caller (this file), not at the wrapper
    assert len(record) == 1 and record[0].filename == __file__
    with pytest.warns(DeprecationWarning):
        sr.owes_ensemble(obs, fct, lambda x: x.mean() * 0.0 + 1.0, backend="numpy")
    with pytest.warns(DeprecationWarning):
        sr.vres_ensemble(obs, fct, lambda x: x.mean() * 0.0 + 1.0, backend="numpy")


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_energy_float32(estimator, backend, to_backend_float32, backend_kwargs):
    rng = np.random.default_rng(3)
    obs = to_backend_float32(rng.normal(size=(N, N_VARS)))
    fct = to_backend_float32(rng.normal(size=(N, ENSEMBLE_SIZE, N_VARS)))
    ens_w = to_backend_float32(rng.uniform(0.1, 1.0, (N, ENSEMBLE_SIZE)))
    kwargs = {"estimator": estimator, **backend_kwargs}
    for res in (
        sr.es_ensemble(obs, fct, **kwargs),
        sr.es_ensemble(obs, fct, ens_w=ens_w, **kwargs),
        sr.twes_ensemble(obs, fct, lambda x: x.clip(min=-0.5), **kwargs),
    ):
        assert_inferred(res, backend)
        assert res.dtype == fct.dtype


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_energy_grad_torch(estimator):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(4)
    obs = torch.as_tensor(rng.normal(size=(4, N_VARS)))
    fct = torch.tensor(rng.normal(size=(4, ENSEMBLE_SIZE, N_VARS)), requires_grad=True)
    ens_w = torch.as_tensor(rng.uniform(0.1, 1.0, (4, ENSEMBLE_SIZE)))
    for kwargs in ({}, {"ens_w": ens_w}):
        fct.grad = None
        sr.es_ensemble(obs, fct, estimator=estimator, **kwargs).sum().backward()
        assert fct.grad is not None and torch.isfinite(fct.grad).all()


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_energy_ens_w_silent_layout(estimator, native, backend_kwargs):
    """fct (D, M, N) with M == N, m_axis=1, v_axis=0: ens_w (M, N) used to be
    moved along the wrong axis without an error, so the score used the wrong
    weights. Compare with the default layout on moveaxis'd inputs."""
    rng = np.random.default_rng(5)
    D, M = 3, 5
    obs = rng.normal(size=(M, D))  # (N, D) with N == M
    fct = rng.normal(size=(M, M, D))  # (N, M, D)
    ens_w = rng.uniform(0.1, 2.0, size=(M, M))  # (N, M)
    fn = native(sr.es_ensemble)
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
