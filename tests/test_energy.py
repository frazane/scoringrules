import numpy as np
import pytest
import scoringrules as sr

from ._parity import MV_CASES, mark_known_mv_failures, mv_ensemble_inputs, mv_v_func


ENSEMBLE_SIZE = 11
N = 20
N_VARS = 3

ESTIMATORS = ["nrg", "fair", "akr", "akr_circperm"]


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_energy_score(estimator, backend):
    # test exceptions

    # incorrect dimensions
    with pytest.raises(ValueError):
        obs = np.random.randn(N, N_VARS)
        fct = np.random.randn(N, ENSEMBLE_SIZE, N_VARS - 1)
        sr.es_ensemble(obs, fct, estimator=estimator, backend=backend)

    # undefined estimator
    with pytest.raises(ValueError):
        fct = np.random.randn(N, ENSEMBLE_SIZE, N_VARS)
        est = "undefined_estimator"
        sr.es_ensemble(obs, fct, estimator=est, backend=backend)

    # test output

    # shapes
    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)
    res = sr.es_ensemble(obs, fct, estimator=estimator, backend=backend)
    assert res.shape == (N,)

    # test equivalence with and without weights
    w = np.ones(fct.shape[:-1])
    res_nrg_w = sr.es_ensemble(obs, fct, ens_w=w, estimator=estimator, backend=backend)
    res_nrg_now = sr.es_ensemble(obs, fct, estimator=estimator, backend=backend)
    assert np.allclose(res_nrg_w, res_nrg_now)

    # correctness
    fct = np.array(
        [
            [0.79546742, 0.4777960, 0.2164079, 0.5409873],
            [0.02461368, 0.7584595, 0.3181810, 0.1319422],
        ]
    ).T
    obs = np.array([0.2743836, 0.8146400])

    res = sr.es_ensemble(obs, fct, estimator=estimator, backend=backend)
    if estimator == "nrg":
        expected = 0.3949793
        np.testing.assert_allclose(res, expected, atol=1e-6)
    elif estimator == "fair":
        expected = 0.3274585
        np.testing.assert_allclose(res, expected, atol=1e-6)
    elif estimator == "akr":
        expected = 0.3522818
        np.testing.assert_allclose(res, expected, atol=1e-6)
    elif estimator == "akr_circperm":
        expected = 0.2778118
        np.testing.assert_allclose(res, expected, atol=1e-6)

    # correctness with weights
    w = np.array([0.1, 0.2, 0.3, 0.4])

    res = sr.es_ensemble(obs, fct, ens_w=w, estimator=estimator, backend=backend)
    if estimator == "nrg":
        expected = 0.4074382
        np.testing.assert_allclose(res, expected, atol=1e-6)
    elif estimator == "fair":
        expected = 0.333500969
        np.testing.assert_allclose(res, expected, atol=1e-6)
    elif estimator == "akr":
        expected = 0.3345371
        np.testing.assert_allclose(res, expected, atol=1e-6)
    elif estimator == "akr_circperm":
        expected = 0.2612048
        np.testing.assert_allclose(res, expected, atol=1e-6)

    # check invariance to scaling of the weights
    w_scaled = w * 10
    res_scaled = sr.es_ensemble(
        obs, fct, ens_w=w_scaled, estimator=estimator, backend=backend
    )
    np.testing.assert_allclose(res_scaled, res, atol=1e-6)


OMIT_ESTIMATORS = ["nrg", "fair"]


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_energy_score_nan_policy(estimator, backend):
    """es_ensemble with NaN forecast members, no user weights."""
    kwargs = {"estimator": estimator, "backend": backend}

    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)

    fct_nan = fct.copy()
    fct_nan[0, [0, 3, 6], 0] = np.nan
    fct_nan[2, [5], 1] = np.nan
    nan_positions = np.isnan(fct_nan).any(axis=(1, 2))

    # default + explicit 'propagate': corrupted samples return NaN
    for policy in (None, "propagate"):
        pol_kwargs = kwargs if policy is None else {**kwargs, "nan_policy": policy}
        res = np.asarray(sr.es_ensemble(obs, fct_nan, **pol_kwargs))
        assert np.all(np.isnan(res[nan_positions]))
        assert not np.any(np.isnan(res[~nan_positions]))

    # 'raise'
    with pytest.raises(ValueError):
        sr.es_ensemble(obs, fct_nan, nan_policy="raise", **kwargs)

    # 'omit' unsupported for akr / akr_circperm
    if estimator not in OMIT_ESTIMATORS:
        with pytest.raises(NotImplementedError):
            sr.es_ensemble(obs, fct_nan, nan_policy="omit", **kwargs)
        return

    # 'omit': no NaN in result
    res = np.asarray(sr.es_ensemble(obs, fct_nan, nan_policy="omit", **kwargs))
    assert not np.any(np.isnan(res))

    # 'omit': equivalence with the surviving sub-ensemble, per row
    for i in range(fct.shape[0]):
        valid = ~np.isnan(fct_nan[i]).any(axis=-1)
        fct_clean = fct_nan[i][valid]
        res_clean = sr.es_ensemble(obs[i], fct_clean, **kwargs)
        res_omit = sr.es_ensemble(obs[i], fct_nan[i], nan_policy="omit", **kwargs)
        assert np.allclose(res_omit, res_clean)

    # all-members-NaN row: 'omit' yields NaN (documented 0/0 edge case)
    fct_allnan = fct.copy()
    fct_allnan[3, :, 0] = np.nan
    res_allnan = np.asarray(
        sr.es_ensemble(obs, fct_allnan, nan_policy="omit", **kwargs)
    )
    assert np.isnan(res_allnan[3])

    # NaN in obs always propagates, even under 'omit'
    obs_nan = obs.copy()
    obs_nan[1, 0] = np.nan
    res = np.asarray(sr.es_ensemble(obs_nan, fct, nan_policy="omit", **kwargs))
    assert np.isnan(res[1])


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_energy_score_w_ens_nan_policy(estimator, backend):
    """es_ensemble with NaN forecast members AND user-supplied ens_w."""
    kwargs = {"estimator": estimator, "backend": backend}

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
        res = np.asarray(
            sr.es_ensemble(obs, fct_nan, ens_w=uniform_ens_w, **pol_kwargs)
        )
        assert np.all(np.isnan(res[nan_positions]))

    # 'raise'
    with pytest.raises(ValueError):
        sr.es_ensemble(obs, fct_nan, ens_w=uniform_ens_w, nan_policy="raise", **kwargs)

    # 'omit' unsupported for akr / akr_circperm
    if estimator not in OMIT_ESTIMATORS:
        with pytest.raises(NotImplementedError):
            sr.es_ensemble(
                obs, fct_nan, ens_w=uniform_ens_w, nan_policy="omit", **kwargs
            )
        return

    # 'omit' with uniform weights: no NaN
    res = np.asarray(
        sr.es_ensemble(obs, fct_nan, ens_w=uniform_ens_w, nan_policy="omit", **kwargs)
    )
    assert not np.any(np.isnan(res))

    # 'omit' with non-uniform weights: weights actually matter
    res = np.asarray(
        sr.es_ensemble(obs, fct, ens_w=non_uniform_ens_w, nan_policy="omit", **kwargs)
    )
    res_nans = np.asarray(
        sr.es_ensemble(
            obs, fct_nan, ens_w=non_uniform_ens_w, nan_policy="omit", **kwargs
        )
    )
    assert not np.any(np.isnan(res_nans))
    assert not np.allclose(res[nan_positions], res_nans[nan_positions])
    assert np.allclose(res[~nan_positions], res_nans[~nan_positions])

    # 'omit': equivalence with clean weighted sub-ensemble, per row
    for i in range(fct.shape[0]):
        valid = ~np.isnan(fct_nan[i]).any(axis=-1)
        res_clean = sr.es_ensemble(
            obs[i],
            fct_nan[i][valid],
            ens_w=uniform_ens_w[i][valid],
            nan_policy="omit",
            **kwargs,
        )
        res_omit = sr.es_ensemble(
            obs[i], fct_nan[i], ens_w=uniform_ens_w[i], nan_policy="omit", **kwargs
        )
        assert np.allclose(res_omit, res_clean)

    # 'omit' with non-uniform weights: equivalence with clean weighted sub-ensemble
    for i in range(fct.shape[0]):
        valid = ~np.isnan(fct_nan[i]).any(axis=-1)
        res_clean = sr.es_ensemble(
            obs[i],
            fct_nan[i][valid],
            ens_w=non_uniform_ens_w[i][valid],
            nan_policy="omit",
            **kwargs,
        )
        res_omit = sr.es_ensemble(
            obs[i], fct_nan[i], ens_w=non_uniform_ens_w[i], nan_policy="omit", **kwargs
        )
        assert np.allclose(res_omit, res_clean)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_energy_score_nan_weights(estimator, backend):
    """es_ensemble when ens_w itself contains NaN (forecasts are clean)."""
    kwargs = {"estimator": estimator, "backend": backend}

    obs = np.random.randn(N, N_VARS)
    fct = np.expand_dims(obs, axis=-2) + np.random.randn(N, ENSEMBLE_SIZE, N_VARS)
    ens_w = np.random.rand(N, ENSEMBLE_SIZE)
    ens_w[0, [0, 3, 6]] = np.nan
    ens_w[2, [5]] = np.nan
    nan_positions = np.isnan(ens_w).any(axis=1)

    # 'propagate'
    res = np.asarray(
        sr.es_ensemble(obs, fct, ens_w=ens_w, nan_policy="propagate", **kwargs)
    )
    assert np.all(np.isnan(res[nan_positions]))

    # 'raise'
    with pytest.raises(ValueError):
        sr.es_ensemble(obs, fct, ens_w=ens_w, nan_policy="raise", **kwargs)

    # 'omit' unsupported for akr / akr_circperm
    if estimator not in OMIT_ESTIMATORS:
        with pytest.raises(NotImplementedError):
            sr.es_ensemble(obs, fct, ens_w=ens_w, nan_policy="omit", **kwargs)
        return

    # 'omit': NaN-weighted members get zero weight (dropped)
    res = np.asarray(sr.es_ensemble(obs, fct, ens_w=ens_w, nan_policy="omit", **kwargs))
    assert not np.any(np.isnan(res))
    for i in range(fct.shape[0]):
        valid = ~np.isnan(ens_w[i])
        res_clean = sr.es_ensemble(
            obs[i], fct[i][valid], ens_w=ens_w[i][valid], nan_policy="omit", **kwargs
        )
        res_omit = sr.es_ensemble(
            obs[i], fct[i], ens_w=ens_w[i], nan_policy="omit", **kwargs
        )
        assert np.allclose(res_omit, res_clean)


@pytest.mark.parametrize("case", MV_CASES)
@pytest.mark.parametrize("M", [6, 7])
@pytest.mark.parametrize("weighted", [False, True], ids=["uw", "w"])
@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("score", ["es_ensemble", "twes_ensemble"])
def test_energy_paths_agree(score, estimator, weighted, M, case, paths_agree, request):
    mark_known_mv_failures(request, case, weighted)
    obs, fct, kwargs = mv_ensemble_inputs(case, M, weighted)
    args = (mv_v_func,) if score == "twes_ensemble" else ()
    paths_agree(getattr(sr, score), obs, fct, *args, estimator=estimator, **kwargs)


def test_energy_score_akr_circperm_ens_w_known_value(backend):
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
        obs, fct, ens_w=np.array(w), estimator="akr_circperm", backend=backend
    )
    # rtol for jax, which runs in float32 here
    np.testing.assert_allclose(np.asarray(res), expected, rtol=1e-5)
