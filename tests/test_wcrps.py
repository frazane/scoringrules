import numpy as np
import pytest
import scoringrules as sr
from .conftest import assert_inferred


M = 11
N = 20

ESTIMATORS = ["nrg", "fair", "pwm", "qd", "akr", "akr_circperm"]


def assert_close(result, expected, backend, *, rtol=1e-5, atol=1e-8):
    """Compare a (possibly framework-native) result to an expected value.

    jax runs in float32 in the test environment, so its tolerances are loosened.
    torch runs in float64 and stays strict.
    """
    arr = np.asarray(result)
    if backend == "jax":
        rtol = max(rtol, 1e-4)
        atol = max(atol, 1e-5)
    assert np.allclose(
        arr, np.asarray(expected), rtol=rtol, atol=atol
    ), f"[{backend}] {arr} != {expected}"


def test_owcrps_ensemble(backend, to_backend, backend_kwargs):
    # test shapes
    obs = to_backend(np.random.randn(N))
    res = sr.owcrps_ensemble(
        obs,
        to_backend(np.random.randn(N, M)),
        w_func=lambda x: x * 0.0 + 1.0,
        **backend_kwargs,
    )
    assert_inferred(res, backend)
    assert res.shape == (N,)

    fct = to_backend(np.random.randn(M, N))
    res = sr.owcrps_ensemble(
        obs,
        fct,
        w_func=lambda x: x * 0.0 + 1.0,
        m_axis=0,
        **backend_kwargs,
    )
    assert_inferred(res, backend)


def test_owcrps_ensemble_nan_policy(backend, to_backend, backend_kwargs):
    """Test behavior of scoringrules.owcrps_ensemble with NaN values."""

    kwargs = {**backend_kwargs, "w_func": lambda x: x * 0.0 + 1.0}

    # Build numpy arrays first so numpy-style indexing stays in numpy land
    obs0 = np.random.randn(N)
    fct0 = np.random.randn(N, M)
    fct_nan0 = fct0.copy()
    fct_nan0[0, [0, 3, 6]] = np.nan
    fct_nan0[2, [5]] = np.nan
    nan_positions = np.isnan(fct_nan0).any(axis=1)

    obs = to_backend(obs0)
    fct_nan = to_backend(fct_nan0)

    # default nan policy (propagate)
    res = sr.owcrps_ensemble(obs, fct_nan, **kwargs)
    assert_inferred(res, backend)
    assert np.all(np.isnan(np.asarray(res)[nan_positions]))

    # 'propagate' nan policy
    res = sr.owcrps_ensemble(obs, fct_nan, nan_policy="propagate", **kwargs)
    assert_inferred(res, backend)
    assert np.all(np.isnan(np.asarray(res)[nan_positions]))

    # 'raise' nan policy
    with pytest.raises(ValueError):
        sr.owcrps_ensemble(obs, fct_nan, nan_policy="raise", **kwargs)

    # 'omit' nan policy: no nans in results
    res = sr.owcrps_ensemble(obs, fct_nan, nan_policy="omit", **kwargs)
    assert_inferred(res, backend)
    assert not np.any(np.isnan(np.asarray(res)))

    # 'omit' nan policy: equivalence with clean ensemble
    for i in range(fct0.shape[0]):
        fct_clean = fct_nan0[i, ~np.isnan(fct_nan0[i])]
        res_clean = sr.owcrps_ensemble(
            to_backend(obs0[i : i + 1]), to_backend(fct_clean[None, :]), **kwargs
        )
        res = sr.owcrps_ensemble(
            to_backend(obs0[i : i + 1]),
            to_backend(fct_nan0[i : i + 1]),
            nan_policy="omit",
            **kwargs,
        )
        assert np.allclose(np.asarray(res), np.asarray(res_clean))


def test_vrcrps_ensemble(backend, to_backend, backend_kwargs):
    # test shapes
    obs = to_backend(np.random.randn(N))
    res = sr.vrcrps_ensemble(
        obs,
        to_backend(np.random.randn(N, M)),
        w_func=lambda x: x * 0.0 + 1.0,
        **backend_kwargs,
    )
    assert_inferred(res, backend)
    assert res.shape == (N,)
    res = sr.vrcrps_ensemble(
        obs,
        to_backend(np.random.randn(M, N)),
        w_func=lambda x: x * 0.0 + 1.0,
        m_axis=0,
        **backend_kwargs,
    )
    assert_inferred(res, backend)
    assert res.shape == (N,)


def test_vrcrps_ensemble_nan_policy(backend, to_backend, backend_kwargs):
    """Test behavior of scoringrules.vrcrps_ensemble with NaN values."""

    kwargs = {**backend_kwargs, "w_func": lambda x: x * 0.0 + 1.0}

    obs0 = np.random.randn(N)
    fct0 = np.random.randn(N, M)
    fct_nan0 = fct0.copy()
    fct_nan0[0, [0, 3, 6]] = np.nan
    fct_nan0[2, [5]] = np.nan
    nan_positions = np.isnan(fct_nan0).any(axis=1)

    obs = to_backend(obs0)
    fct_nan = to_backend(fct_nan0)

    # default nan policy (propagate)
    res = sr.vrcrps_ensemble(obs, fct_nan, **kwargs)
    assert_inferred(res, backend)
    assert np.all(np.isnan(np.asarray(res)[nan_positions]))

    # 'propagate' nan policy
    res = sr.vrcrps_ensemble(obs, fct_nan, nan_policy="propagate", **kwargs)
    assert_inferred(res, backend)
    assert np.all(np.isnan(np.asarray(res)[nan_positions]))

    # 'raise' nan policy
    with pytest.raises(ValueError):
        sr.vrcrps_ensemble(obs, fct_nan, nan_policy="raise", **kwargs)

    # 'omit' nan policy: no nans in results
    res = sr.vrcrps_ensemble(obs, fct_nan, nan_policy="omit", **kwargs)
    assert_inferred(res, backend)
    assert not np.any(np.isnan(np.asarray(res)))

    # 'omit' nan policy: equivalence with clean ensemble
    for i in range(fct0.shape[0]):
        fct_clean = fct_nan0[i, ~np.isnan(fct_nan0[i])]
        res_clean = sr.vrcrps_ensemble(
            to_backend(obs0[i : i + 1]), to_backend(fct_clean[None, :]), **kwargs
        )
        res = sr.vrcrps_ensemble(
            to_backend(obs0[i : i + 1]),
            to_backend(fct_nan0[i : i + 1]),
            nan_policy="omit",
            **kwargs,
        )
        assert np.allclose(np.asarray(res), np.asarray(res_clean))


def test_owcrps_ensemble_nan_weights(backend, to_backend, backend_kwargs):
    """Test owcrps_ensemble when the ensemble weights (ens_w) contain NaN."""
    kwargs = {**backend_kwargs, "w_func": lambda x: x * 0.0 + 1.0}

    obs0 = np.random.randn(N)
    fct0 = np.random.randn(N, M)
    ens_w0 = np.random.rand(N, M)
    ens_w0[0, [0, 3, 6]] = np.nan
    ens_w0[2, [5]] = np.nan
    nan_positions = np.isnan(ens_w0).any(axis=1)

    obs = to_backend(obs0)
    fct = to_backend(fct0)
    ens_w = to_backend(ens_w0)

    # 'propagate': a NaN weight yields NaN output
    res = sr.owcrps_ensemble(obs, fct, ens_w=ens_w, nan_policy="propagate", **kwargs)
    assert_inferred(res, backend)
    assert np.all(np.isnan(np.asarray(res)[nan_positions]))

    # 'raise': error if a weight is NaN
    with pytest.raises(ValueError):
        sr.owcrps_ensemble(obs, fct, ens_w=ens_w, nan_policy="raise", **kwargs)

    # 'omit': NaN-weighted members get zero weight (equivalent to dropping them)
    res = sr.owcrps_ensemble(obs, fct, ens_w=ens_w, nan_policy="omit", **kwargs)
    assert_inferred(res, backend)
    res_np = np.asarray(res)
    assert not np.any(np.isnan(res_np))
    for i in range(fct0.shape[0]):
        valid = ~np.isnan(ens_w0[i])
        res_clean = sr.owcrps_ensemble(
            to_backend(obs0[i : i + 1]),
            to_backend(fct0[i : i + 1, valid]),
            ens_w=to_backend(ens_w0[i : i + 1, valid]),
            nan_policy="omit",
            **kwargs,
        )
        assert np.allclose(res_np[i], np.asarray(res_clean))


def test_vrcrps_ensemble_nan_weights(backend, to_backend, backend_kwargs):
    """Test vrcrps_ensemble when the ensemble weights (ens_w) contain NaN."""
    kwargs = {**backend_kwargs, "w_func": lambda x: x * 0.0 + 1.0}

    obs0 = np.random.randn(N)
    fct0 = np.random.randn(N, M)
    ens_w0 = np.random.rand(N, M)
    ens_w0[0, [0, 3, 6]] = np.nan
    ens_w0[2, [5]] = np.nan
    nan_positions = np.isnan(ens_w0).any(axis=1)

    obs = to_backend(obs0)
    fct = to_backend(fct0)
    ens_w = to_backend(ens_w0)

    # 'propagate': a NaN weight yields NaN output
    res = sr.vrcrps_ensemble(obs, fct, ens_w=ens_w, nan_policy="propagate", **kwargs)
    assert_inferred(res, backend)
    assert np.all(np.isnan(np.asarray(res)[nan_positions]))

    # 'raise': error if a weight is NaN
    with pytest.raises(ValueError):
        sr.vrcrps_ensemble(obs, fct, ens_w=ens_w, nan_policy="raise", **kwargs)

    # 'omit': NaN-weighted members get zero weight (equivalent to dropping them)
    res = sr.vrcrps_ensemble(obs, fct, ens_w=ens_w, nan_policy="omit", **kwargs)
    assert_inferred(res, backend)
    res_np = np.asarray(res)
    assert not np.any(np.isnan(res_np))
    for i in range(fct0.shape[0]):
        valid = ~np.isnan(ens_w0[i])
        res_clean = sr.vrcrps_ensemble(
            to_backend(obs0[i : i + 1]),
            to_backend(fct0[i : i + 1, valid]),
            ens_w=to_backend(ens_w0[i : i + 1, valid]),
            nan_policy="omit",
            **kwargs,
        )
        assert np.allclose(res_np[i], np.asarray(res_clean))


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_twcrps_vs_crps(estimator, backend, to_backend, backend_kwargs):
    obs0 = np.random.randn(N)
    mu = obs0 + np.random.randn(N) * 0.1
    sigma = abs(np.random.randn(N)) * 0.3
    fct0 = np.random.randn(N, M) * sigma[..., None] + mu[..., None]

    obs = to_backend(obs0)
    fct = to_backend(fct0)

    res = sr.crps_ensemble(obs, fct, estimator=estimator, **backend_kwargs)
    assert_inferred(res, backend)

    # no argument given
    resw = sr.twcrps_ensemble(obs, fct, estimator=estimator, **backend_kwargs)
    assert_inferred(resw, backend)
    assert_close(res, resw, backend, rtol=1e-10)

    # a and b
    resw = sr.twcrps_ensemble(
        obs, fct, a=float("-inf"), b=float("inf"), estimator=estimator, **backend_kwargs
    )
    assert_inferred(resw, backend)
    assert_close(res, resw, backend, rtol=1e-10)

    # v_func as identity function
    resw = sr.twcrps_ensemble(
        obs, fct, v_func=lambda x: x, estimator=estimator, **backend_kwargs
    )
    assert_inferred(resw, backend)
    assert_close(res, resw, backend, rtol=1e-10)


def test_owcrps_vs_crps(backend, to_backend, backend_kwargs):
    obs0 = np.random.randn(N)
    mu = obs0 + np.random.randn(N) * 0.1
    sigma = abs(np.random.randn(N)) * 0.5
    fct0 = np.random.randn(N, M) * sigma[..., None] + mu[..., None]

    obs = to_backend(obs0)
    fct = to_backend(fct0)

    res = sr.crps_ensemble(obs, fct, estimator="qd", **backend_kwargs)
    assert_inferred(res, backend)

    # no argument given
    resw = sr.owcrps_ensemble(obs, fct, **backend_kwargs)
    assert_inferred(resw, backend)
    assert_close(res, resw, backend, rtol=1e-4)

    # a and b
    resw = sr.owcrps_ensemble(
        obs, fct, a=float("-inf"), b=float("inf"), **backend_kwargs
    )
    assert_inferred(resw, backend)
    assert_close(res, resw, backend, rtol=1e-4)

    # w_func as identity function
    resw = sr.owcrps_ensemble(
        obs, fct, w_func=lambda x: x * 0.0 + 1.0, **backend_kwargs
    )
    assert_inferred(resw, backend)
    assert_close(res, resw, backend, rtol=1e-4)


def test_vrcrps_vs_crps(backend, to_backend, backend_kwargs):
    obs0 = np.random.randn(N)
    mu = obs0 + np.random.randn(N) * 0.1
    sigma = abs(np.random.randn(N)) * 0.3
    fct0 = np.random.randn(N, M) * sigma[..., None] + mu[..., None]

    obs = to_backend(obs0)
    fct = to_backend(fct0)

    res = sr.crps_ensemble(obs, fct, estimator="nrg", **backend_kwargs)
    assert_inferred(res, backend)

    # no argument given
    resw = sr.vrcrps_ensemble(obs, fct, **backend_kwargs)
    assert_inferred(resw, backend)
    assert_close(res, resw, backend, rtol=1e-5)

    # a and b
    resw = sr.vrcrps_ensemble(
        obs, fct, a=float("-inf"), b=float("inf"), **backend_kwargs
    )
    assert_inferred(resw, backend)
    assert_close(res, resw, backend, rtol=1e-5)

    # w_func as identity function
    resw = sr.vrcrps_ensemble(
        obs, fct, w_func=lambda x: x * 0.0 + 1.0, **backend_kwargs
    )
    assert_inferred(resw, backend)
    assert_close(res, resw, backend, rtol=1e-5)


def test_owcrps_score_correctness(backend, to_backend, backend_kwargs):
    fct0 = np.array(
        [
            [-0.03574194, 0.06873582, 0.03098684, 0.07316138, 0.08498165],
            [-0.11957874, 0.26472238, -0.06324622, 0.43026451, -0.25640457],
            [-1.31340831, -1.43722153, -1.01696021, -0.70790148, -1.20432392],
            [1.26054027, 1.03477874, 0.61688454, 0.75305795, 1.19364529],
            [0.63192933, 0.33521695, 0.20626017, 0.43158264, 0.69518928],
            [0.83305233, 0.71965134, 0.96687378, 1.0000473, 0.82714425],
            [0.0128655, 0.03849841, 0.02459106, -0.02618909, 0.18687008],
            [-0.69399216, -0.59432627, -1.43629972, 0.01979857, -0.3286767],
            [1.92958764, 2.2678255, 1.86036922, 1.84766384, 2.03331138],
            [0.80847492, 1.11265193, 0.58995365, 1.04838184, 1.10439277],
        ]
    )

    obs0 = np.array(
        [
            0.19640722,
            -0.11300369,
            -1.0400268,
            0.84306533,
            0.36572078,
            0.82487264,
            0.14088773,
            -0.51936113,
            1.70706264,
            0.75985908,
        ]
    )

    fct = to_backend(fct0)
    obs = to_backend(obs0)
    rtol = 1e-4 if backend == "jax" else 1e-6

    def w_func(x):
        return (x > -1) * 1.0

    res = sr.owcrps_ensemble(obs, fct, w_func=w_func, **backend_kwargs)
    assert_inferred(res, backend)
    np.testing.assert_allclose(np.mean(np.asarray(res)), 0.09320807, rtol=rtol)

    res = sr.owcrps_ensemble(obs, fct, a=-1.0, **backend_kwargs)
    assert_inferred(res, backend)
    np.testing.assert_allclose(np.mean(np.asarray(res)), 0.09320807, rtol=rtol)

    def w_func(x):
        return (x < 1.85) * 1.0

    res = sr.owcrps_ensemble(obs, fct, w_func=w_func, **backend_kwargs)
    assert_inferred(res, backend)
    np.testing.assert_allclose(np.mean(np.asarray(res)), 0.09933139, rtol=rtol)

    res = sr.owcrps_ensemble(obs, fct, b=1.85, **backend_kwargs)
    assert_inferred(res, backend)
    np.testing.assert_allclose(np.mean(np.asarray(res)), 0.09933139, rtol=rtol)

    # test equivalence with and without weights
    w = to_backend(np.ones(fct0.shape))
    res_nrg_w = sr.owcrps_ensemble(obs, fct, ens_w=w, b=1.85, **backend_kwargs)
    res_nrg_now = sr.owcrps_ensemble(obs, fct, b=1.85, **backend_kwargs)
    assert_inferred(res_nrg_w, backend)
    assert_inferred(res_nrg_now, backend)
    atol = 1e-4 if backend == "jax" else 1e-8
    assert np.allclose(np.asarray(res_nrg_w), np.asarray(res_nrg_now), atol=atol)


def test_twcrps_score_correctness(backend, to_backend, backend_kwargs):
    fct0 = np.array(
        [
            [-0.03574194, 0.06873582, 0.03098684, 0.07316138, 0.08498165],
            [-0.11957874, 0.26472238, -0.06324622, 0.43026451, -0.25640457],
            [-1.31340831, -1.43722153, -1.01696021, -0.70790148, -1.20432392],
            [1.26054027, 1.03477874, 0.61688454, 0.75305795, 1.19364529],
            [0.63192933, 0.33521695, 0.20626017, 0.43158264, 0.69518928],
            [0.83305233, 0.71965134, 0.96687378, 1.0000473, 0.82714425],
            [0.0128655, 0.03849841, 0.02459106, -0.02618909, 0.18687008],
            [-0.69399216, -0.59432627, -1.43629972, 0.01979857, -0.3286767],
            [1.92958764, 2.2678255, 1.86036922, 1.84766384, 2.03331138],
            [0.80847492, 1.11265193, 0.58995365, 1.04838184, 1.10439277],
        ]
    )

    obs0 = np.array(
        [
            0.19640722,
            -0.11300369,
            -1.0400268,
            0.84306533,
            0.36572078,
            0.82487264,
            0.14088773,
            -0.51936113,
            1.70706264,
            0.75985908,
        ]
    )

    fct = to_backend(fct0)
    obs = to_backend(obs0)
    rtol = 1e-4 if backend == "jax" else 1e-6

    def v_func(x):
        from scoringrules.backend import get_namespace

        xp = get_namespace(x)
        return xp.maximum(x, xp.asarray(-1.0))

    res = sr.twcrps_ensemble(obs, fct, v_func=v_func, estimator="nrg", **backend_kwargs)
    assert_inferred(res, backend)
    np.testing.assert_allclose(np.mean(np.asarray(res)), 0.09489662, rtol=rtol)

    res = sr.twcrps_ensemble(obs, fct, a=-1.0, estimator="nrg", **backend_kwargs)
    assert_inferred(res, backend)
    np.testing.assert_allclose(np.mean(np.asarray(res)), 0.09489662, rtol=rtol)

    def v_func(x):
        from scoringrules.backend import get_namespace

        xp = get_namespace(x)
        return xp.minimum(x, xp.asarray(1.85))

    res = sr.twcrps_ensemble(obs, fct, v_func=v_func, estimator="nrg", **backend_kwargs)
    assert_inferred(res, backend)
    np.testing.assert_allclose(np.mean(np.asarray(res)), 0.0994809, rtol=rtol)

    res = sr.twcrps_ensemble(obs, fct, b=1.85, estimator="nrg", **backend_kwargs)
    assert_inferred(res, backend)
    np.testing.assert_allclose(np.mean(np.asarray(res)), 0.0994809, rtol=rtol)


def test_vrcrps_score_correctness(backend, to_backend, backend_kwargs):
    fct0 = np.array(
        [
            [-0.03574194, 0.06873582, 0.03098684, 0.07316138, 0.08498165],
            [-0.11957874, 0.26472238, -0.06324622, 0.43026451, -0.25640457],
            [-1.31340831, -1.43722153, -1.01696021, -0.70790148, -1.20432392],
            [1.26054027, 1.03477874, 0.61688454, 0.75305795, 1.19364529],
            [0.63192933, 0.33521695, 0.20626017, 0.43158264, 0.69518928],
            [0.83305233, 0.71965134, 0.96687378, 1.0000473, 0.82714425],
            [0.0128655, 0.03849841, 0.02459106, -0.02618909, 0.18687008],
            [-0.69399216, -0.59432627, -1.43629972, 0.01979857, -0.3286767],
            [1.92958764, 2.2678255, 1.86036922, 1.84766384, 2.03331138],
            [0.80847492, 1.11265193, 0.58995365, 1.04838184, 1.10439277],
        ]
    )

    obs0 = np.array(
        [
            0.19640722,
            -0.11300369,
            -1.0400268,
            0.84306533,
            0.36572078,
            0.82487264,
            0.14088773,
            -0.51936113,
            1.70706264,
            0.75985908,
        ]
    )

    fct = to_backend(fct0)
    obs = to_backend(obs0)
    rtol = 1e-4 if backend == "jax" else 1e-6

    def w_func(x):
        return (x > -1) * 1.0

    res = sr.vrcrps_ensemble(obs, fct, w_func=w_func, **backend_kwargs)
    assert_inferred(res, backend)
    np.testing.assert_allclose(np.mean(np.asarray(res)), 0.1003983, rtol=rtol)

    res = sr.vrcrps_ensemble(obs, fct, a=-1.0, **backend_kwargs)
    assert_inferred(res, backend)
    np.testing.assert_allclose(np.mean(np.asarray(res)), 0.1003983, rtol=rtol)

    def w_func(x):
        return (x < 1.85) * 1.0

    res = sr.vrcrps_ensemble(obs, fct, w_func=w_func, **backend_kwargs)
    assert_inferred(res, backend)
    np.testing.assert_allclose(np.mean(np.asarray(res)), 0.1950857, rtol=rtol)

    res = sr.vrcrps_ensemble(obs, fct, b=1.85, **backend_kwargs)
    assert_inferred(res, backend)
    np.testing.assert_allclose(np.mean(np.asarray(res)), 0.1950857, rtol=rtol)
