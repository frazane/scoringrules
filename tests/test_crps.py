import numpy as np
import pytest
import scipy.stats as st
import scoringrules as sr
from .conftest import assert_inferred

ENSEMBLE_SIZE = 11
N = 5

ESTIMATORS = ["nrg", "fair", "pwm", "int", "qd", "akr", "akr_circperm"]


def assert_close(result, expected, backend, *, rtol=1e-5, atol=1e-8):
    """Compare a (possibly framework-native) result to an expected numpy/scalar value.

    jax runs in float32 in the test environment (``jax_enable_x64`` is off), so its
    tolerances are loosened. torch runs in float64 and stays strict.
    """
    arr = np.asarray(result)
    if backend == "jax":
        rtol = max(rtol, 1e-4)
        atol = max(atol, 1e-5)
    assert np.allclose(
        arr, np.asarray(expected), rtol=rtol, atol=atol
    ), f"[{backend}] {arr} != {expected}"


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_crps_ensemble(estimator, backend, to_backend, backend_kwargs):
    """Test general behavior of scoringrules.crps_ensemble."""

    kwargs = {"estimator": estimator, **backend_kwargs}

    # test data
    obs0 = np.random.randn(N)
    fct0 = np.random.randn(N, ENSEMBLE_SIZE)
    obs = to_backend(obs0)
    fct = to_backend(fct0)

    # test exceptions: undefined estimator
    with pytest.raises(ValueError):
        sr.crps_ensemble(obs, fct, estimator="undefined_estimator", **backend_kwargs)

    # test shapes: default case
    res = sr.crps_ensemble(obs, fct, **kwargs)
    assert_inferred(res, backend)
    res = np.asarray(res)
    res_mean = np.mean(res)
    assert res.shape == (N,)
    # test shapes: with extra dimension
    res = sr.crps_ensemble(to_backend(obs0[None]), to_backend(fct0[None, :]), **kwargs)
    assert_inferred(res, backend)
    res = np.asarray(res)
    assert res.shape == (1, N)
    if backend == "jax":
        assert np.allclose(np.mean(res), res_mean, rtol=1e-5)
    else:
        assert np.mean(res) == res_mean
    # test shapes: with non-default ensemble axis
    res = sr.crps_ensemble(
        to_backend(obs0[..., None]), to_backend(fct0[..., None]), m_axis=-2, **kwargs
    )
    assert_inferred(res, backend)
    res = np.asarray(res)
    assert res.shape == (N, 1)
    if backend == "jax":
        assert np.allclose(np.mean(res), res_mean, rtol=1e-5)
    else:
        assert np.mean(res) == res_mean

    # non-negative values
    if estimator not in ["akr", "akr_circperm"]:
        res = sr.crps_ensemble(obs, fct, **kwargs)
        assert_inferred(res, backend)
        res = np.asarray(res)
        assert not np.any(res < 0.0)

    # approx zero when perfect forecast
    perfect_fct = obs0[..., None] + np.random.randn(N, ENSEMBLE_SIZE) * 0.00001
    res = sr.crps_ensemble(obs, to_backend(perfect_fct), **kwargs)
    assert_inferred(res, backend)
    res = np.asarray(res)
    assert not np.any(res - 0.0 > 0.0001)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_crps_ensemble_nan_policy(estimator, backend, to_backend, backend_kwargs):
    """Test behavior of scoringrules.crps_ensemble with NaN values."""

    kwargs = {"estimator": estimator, **backend_kwargs}

    # test data
    obs0 = np.random.randn(N)
    fct0 = np.random.randn(N, ENSEMBLE_SIZE)

    fct_nan0 = fct0.copy()
    fct_nan0[0, [0, 3, 6]] = np.nan
    fct_nan0[2, [5]] = np.nan
    nan_positions = np.isnan(fct_nan0).any(axis=1)

    obs = to_backend(obs0)
    fct_nan = to_backend(fct_nan0)

    # default nan policy (propagate)
    res = sr.crps_ensemble(obs, fct_nan, **kwargs)
    assert_inferred(res, backend)
    assert np.all(np.isnan(np.asarray(res)[nan_positions]))

    # 'propagate' nan policy
    res = sr.crps_ensemble(obs, fct_nan, nan_policy="propagate", **kwargs)
    assert_inferred(res, backend)
    assert np.all(np.isnan(np.asarray(res)[nan_positions]))

    # 'raise' nan policy
    with pytest.raises(ValueError):
        sr.crps_ensemble(obs, fct_nan, nan_policy="raise", **kwargs)

    # 'omit' nan policy: exceptions for int, akr, akr_circperm estimators
    if estimator in ["int", "akr", "akr_circperm"]:
        with pytest.raises(NotImplementedError):
            sr.crps_ensemble(obs, fct_nan, nan_policy="omit", **kwargs)
        return

    # 'omit' nan policy: no nans in results
    res = sr.crps_ensemble(obs, fct_nan, nan_policy="omit", **kwargs)
    assert_inferred(res, backend)
    assert not np.any(np.isnan(np.asarray(res)))

    # 'omit' nan policy: equivalence with clean ensemble
    rtol = 1e-4 if backend == "jax" else 1e-5
    for i in range(fct0.shape[0]):
        fct_clean = fct_nan0[i, ~np.isnan(fct_nan0[i])]
        res_clean = sr.crps_ensemble(
            to_backend(obs0[i : i + 1]), to_backend(fct_clean[None, :]), **kwargs
        )
        res = sr.crps_ensemble(
            to_backend(obs0[i : i + 1]),
            to_backend(fct_nan0[i : i + 1]),
            nan_policy="omit",
            **kwargs,
        )
        assert np.allclose(np.asarray(res), np.asarray(res_clean), rtol=rtol)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_crps_ensemble_w_ens(estimator, backend, to_backend, backend_kwargs):
    """Test behavior of scoringrules.crps_ensemble with ensemble weights."""
    kwargs = {"estimator": estimator, **backend_kwargs}

    # test data
    obs0 = np.random.randn(N)
    mu = obs0 + np.random.randn(N) * 0.3
    sigma = abs(np.random.randn(N)) * 0.5
    fct0 = np.random.randn(N, ENSEMBLE_SIZE) * sigma[..., None] + mu[..., None]
    obs = to_backend(obs0)
    fct = to_backend(fct0)
    uniform_ens_w = to_backend(np.ones(fct0.shape))
    non_uniform_ens_w = to_backend(np.random.rand(*fct0.shape))

    # default
    res = sr.crps_ensemble(obs, fct, **kwargs)
    assert_inferred(res, backend)

    # equivalence for uniform weights
    res_uniform_weights = sr.crps_ensemble(obs, fct, ens_w=uniform_ens_w, **kwargs)
    assert_inferred(res_uniform_weights, backend)
    atol = 1e-4 if backend == "jax" else 1e-6
    assert np.allclose(np.asarray(res_uniform_weights), np.asarray(res), atol=atol)

    # non-equivalence for non-uniform weights
    res_non_uniform_weights = sr.crps_ensemble(
        obs, fct, ens_w=non_uniform_ens_w, **kwargs
    )
    assert_inferred(res_non_uniform_weights, backend)
    assert not np.allclose(
        np.asarray(res_non_uniform_weights), np.asarray(res), atol=1e-6
    )

    # estimator equivalence with weights
    w = to_backend(np.abs(np.random.randn(N, ENSEMBLE_SIZE) * sigma[..., None]))
    res_nrg = sr.crps_ensemble(obs, fct, ens_w=w, estimator="nrg", **backend_kwargs)
    res_qd = sr.crps_ensemble(obs, fct, ens_w=w, estimator="qd", **backend_kwargs)
    assert_inferred(res_nrg, backend)
    assert_inferred(res_qd, backend)
    if backend in ["torch", "jax"]:
        assert np.allclose(np.asarray(res_nrg), np.asarray(res_qd), rtol=1e-03)
    else:
        assert np.allclose(np.asarray(res_nrg), np.asarray(res_qd))

    # correctness against a known value (qd estimator with integer weights);
    # exercised once on the pure-python/numpy path
    if backend == "numpy":
        obs_known = -0.6042506
        fct_known = np.array(
            [
                1.7812118,
                0.5863797,
                0.7038174,
                -0.7743998,
                -0.2751647,
                1.1863249,
                1.2990966,
                -0.3242982,
                -0.5968781,
                0.9064937,
            ]
        )
        res_known = sr.crps_ensemble(
            obs_known, fct_known, ens_w=np.arange(10), estimator="qd"
        )
        assert np.isclose(res_known, 0.4923673)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_crps_ensemble_w_ens_nan_policy(estimator, backend, to_backend, backend_kwargs):
    """Test behavior of scoringrules.crps_ensemble with ensemble weights and NaN values."""

    kwargs = {"estimator": estimator, **backend_kwargs}

    # test data
    obs0 = np.random.randn(N)
    mu = obs0 + np.random.randn(N) * 0.3
    sigma = abs(np.random.randn(N)) * 0.5
    fct0 = np.random.randn(N, ENSEMBLE_SIZE) * sigma[..., None] + mu[..., None]
    uniform_ens_w0 = np.ones(fct0.shape)
    non_uniform_ens_w0 = np.random.rand(*fct0.shape)
    fct_nan0 = fct0.copy()
    fct_nan0[0, [0, 3, 6]] = np.nan
    fct_nan0[2, [5]] = np.nan
    nan_positions = np.isnan(fct_nan0).any(axis=1)

    obs = to_backend(obs0)
    fct = to_backend(fct0)
    fct_nan = to_backend(fct_nan0)
    uniform_ens_w = to_backend(uniform_ens_w0)
    non_uniform_ens_w = to_backend(non_uniform_ens_w0)

    # default nan policy (propagate): ensembles with NaN members return NaN
    res = sr.crps_ensemble(obs, fct_nan, ens_w=uniform_ens_w, **kwargs)
    assert_inferred(res, backend)
    assert np.all(np.isnan(np.asarray(res)[nan_positions]))

    # 'propagate' nan policy: ensembles with NaN members return NaN
    res = sr.crps_ensemble(
        obs, fct_nan, ens_w=uniform_ens_w, nan_policy="propagate", **kwargs
    )
    assert_inferred(res, backend)
    assert np.all(np.isnan(np.asarray(res)[nan_positions]))

    # 'raise' nan policy: error if NaN is encountered
    with pytest.raises(ValueError):
        sr.crps_ensemble(
            obs, fct_nan, ens_w=uniform_ens_w, nan_policy="raise", **kwargs
        )

    # 'omit' nan policy: exceptions for int, akr, akr_circperm estimators
    if estimator in ["int", "akr", "akr_circperm"]:
        with pytest.raises(NotImplementedError):
            sr.crps_ensemble(
                obs, fct_nan, ens_w=uniform_ens_w, nan_policy="omit", **kwargs
            )
        return

    # 'omit' nan policy: no nans in results
    res = sr.crps_ensemble(
        obs, fct_nan, ens_w=uniform_ens_w, nan_policy="omit", **kwargs
    )
    assert_inferred(res, backend)
    assert not np.any(np.isnan(np.asarray(res)))

    # 'omit' nan policy: non-equivalence if non-uniform weights are used
    res = sr.crps_ensemble(
        obs, fct, ens_w=non_uniform_ens_w, nan_policy="omit", **kwargs
    )
    res_nans = sr.crps_ensemble(
        obs, fct_nan, ens_w=non_uniform_ens_w, nan_policy="omit", **kwargs
    )
    assert_inferred(res, backend)
    assert_inferred(res_nans, backend)
    res, res_nans = np.asarray(res), np.asarray(res_nans)
    assert not np.any(np.isnan(res_nans))
    assert not np.allclose(res[nan_positions], res_nans[nan_positions])
    assert np.allclose(res[~nan_positions], res_nans[~nan_positions])

    # 'omit' nan policy: equivalence with clean ensemble
    rtol = 1e-4 if backend == "jax" else 1e-5
    for i in range(fct0.shape[0]):
        valid = ~np.isnan(fct_nan0[i])
        fct_clean = fct_nan0[i, valid]
        uniform_ens_w_clean = uniform_ens_w0[i, valid]
        res_clean = sr.crps_ensemble(
            to_backend(obs0[i]),
            to_backend(fct_clean),
            ens_w=to_backend(uniform_ens_w_clean),
            nan_policy="omit",
            **kwargs,
        )
        res = sr.crps_ensemble(
            to_backend(obs0[i]),
            to_backend(fct_nan0[i]),
            ens_w=to_backend(uniform_ens_w0[i]),
            nan_policy="omit",
            **kwargs,
        )
        assert np.allclose(np.asarray(res), np.asarray(res_clean), rtol=rtol)


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_crps_ensemble_nan_weights(estimator, backend, to_backend, backend_kwargs):
    """Test crps_ensemble when the ensemble weights (ens_w) contain NaN.

    A member is invalid if its forecast value OR its weight is NaN:
    'propagate' -> NaN, 'raise' -> error, 'omit' -> zero weight (dropped).
    """
    kwargs = {"estimator": estimator, **backend_kwargs}

    obs0 = np.random.randn(N)
    fct0 = np.random.randn(N, ENSEMBLE_SIZE)
    ens_w0 = np.random.rand(N, ENSEMBLE_SIZE)
    ens_w0[0, [0, 3, 6]] = np.nan
    ens_w0[2, [5]] = np.nan
    nan_positions = np.isnan(ens_w0).any(axis=1)

    obs = to_backend(obs0)
    fct = to_backend(fct0)
    ens_w = to_backend(ens_w0)

    # 'propagate': a NaN weight yields NaN output
    res = sr.crps_ensemble(obs, fct, ens_w=ens_w, nan_policy="propagate", **kwargs)
    assert_inferred(res, backend)
    assert np.all(np.isnan(np.asarray(res)[nan_positions]))

    # 'raise': error if a weight is NaN
    with pytest.raises(ValueError):
        sr.crps_ensemble(obs, fct, ens_w=ens_w, nan_policy="raise", **kwargs)

    # 'omit': not implemented for int/akr/akr_circperm estimators
    if estimator in ["int", "akr", "akr_circperm"]:
        with pytest.raises(NotImplementedError):
            sr.crps_ensemble(obs, fct, ens_w=ens_w, nan_policy="omit", **kwargs)
        return

    # 'omit': NaN-weighted members get zero weight (equivalent to dropping them)
    res = sr.crps_ensemble(obs, fct, ens_w=ens_w, nan_policy="omit", **kwargs)
    assert_inferred(res, backend)
    res = np.asarray(res)
    assert not np.any(np.isnan(res))
    rtol = 1e-4 if backend == "jax" else 1e-5
    for i in range(fct0.shape[0]):
        valid = ~np.isnan(ens_w0[i])
        res_clean = sr.crps_ensemble(
            to_backend(obs0[i]),
            to_backend(fct0[i, valid]),
            ens_w=to_backend(ens_w0[i, valid]),
            nan_policy="omit",
            **kwargs,
        )
        assert np.allclose(res[i], np.asarray(res_clean), rtol=rtol)

    # the same result is obtained with the ensemble on a non-default axis
    res_axis = sr.crps_ensemble(
        to_backend(obs0[..., None]),
        to_backend(fct0[..., None]),
        ens_w=to_backend(ens_w0[..., None]),
        m_axis=-2,
        nan_policy="omit",
        **kwargs,
    )
    assert_inferred(res_axis, backend)
    assert np.allclose(np.asarray(res_axis).ravel(), res, rtol=rtol)


def test_crps_ensemble_correctness(backend, to_backend, backend_kwargs):
    obs0 = np.random.randn(N)
    mu = obs0 + np.random.randn(N) * 0.3
    sigma = abs(np.random.randn(N)) * 0.5
    fct0 = np.random.randn(N, ENSEMBLE_SIZE) * sigma[..., None] + mu[..., None]
    obs = to_backend(obs0)
    fct = to_backend(fct0)

    # test equivalence of different estimators
    res_nrg = sr.crps_ensemble(obs, fct, estimator="nrg", **backend_kwargs)
    res_qd = sr.crps_ensemble(obs, fct, estimator="qd", **backend_kwargs)
    res_fair = sr.crps_ensemble(obs, fct, estimator="fair", **backend_kwargs)
    res_pwm = sr.crps_ensemble(obs, fct, estimator="pwm", **backend_kwargs)
    for r in (res_nrg, res_qd, res_fair, res_pwm):
        assert_inferred(r, backend)
    res_nrg, res_qd = np.asarray(res_nrg), np.asarray(res_qd)
    res_fair, res_pwm = np.asarray(res_fair), np.asarray(res_pwm)
    if backend in ["torch", "jax"]:
        assert np.allclose(res_nrg, res_qd, rtol=1e-03)
        assert np.allclose(res_fair, res_pwm, rtol=1e-03)
    else:
        assert np.allclose(res_nrg, res_qd)
        assert np.allclose(res_fair, res_pwm)

    # correctness on the pure-python/numpy path (exercised once)
    if backend == "numpy":
        obs_known = -0.6042506
        fct_known = np.array(
            [
                1.7812118,
                0.5863797,
                0.7038174,
                -0.7743998,
                -0.2751647,
                1.1863249,
                1.2990966,
                -0.3242982,
                -0.5968781,
                0.9064937,
            ]
        )
        res = sr.crps_ensemble(obs_known, fct_known, estimator="qd")
        assert np.isclose(res, 0.6126602)


def test_crps_quantile(backend, to_backend, backend_kwargs):
    # test shapes
    obs = np.random.randn(N)
    fct = np.random.randn(N, ENSEMBLE_SIZE)
    alpha = np.linspace(0.1, 0.9, ENSEMBLE_SIZE)
    res = sr.crps_quantile(
        to_backend(obs), to_backend(fct), to_backend(alpha), **backend_kwargs
    )
    assert_inferred(res, backend)
    assert np.asarray(res).shape == (N,)

    fct_t = np.random.randn(ENSEMBLE_SIZE, N)
    res = sr.crps_quantile(
        to_backend(obs),
        to_backend(fct_t),
        to_backend(alpha),
        m_axis=0,
        **backend_kwargs,
    )
    assert_inferred(res, backend)
    assert np.asarray(res).shape == (N,)

    # Test quantile approximation close to analytical normal crps if forecast comes
    # from the normal distribution
    for mu in np.random.sample(size=10):
        for A in [9, 99, 999]:
            a0 = 1 / (A + 1)
            a1 = 1 - a0
            fctq = (
                st.norm(np.repeat(mu, N), np.ones(N))
                .ppf(np.linspace(np.repeat(a0, N), np.repeat(a1, N), A))
                .T
            )
            alphaq = np.linspace(a0, a1, A)
            obsq = np.repeat(mu, N)
            qcrps = sr.crps_quantile(
                to_backend(obsq), to_backend(fctq), to_backend(alphaq), **backend_kwargs
            )
            ncrps = sr.crps_normal(
                to_backend(obsq),
                to_backend(np.repeat(mu, N)),
                to_backend(np.ones(N)),
                **backend_kwargs,
            )
            assert_inferred(qcrps, backend)
            assert_inferred(ncrps, backend)
            percentage_error_to_analytic = 1 - np.asarray(qcrps) / np.asarray(ncrps)
            # jax runs in float32; allow a little extra slack on the tightest grids
            tol = (1 / A) + (1e-3 if backend == "jax" else 0.0)
            assert np.all(
                np.abs(percentage_error_to_analytic) < tol
            ), "Quantile CRPS should be close to normal CRPS"

    # Test raise valueerror if array sizes don't match
    with pytest.raises(ValueError):
        sr.crps_quantile(
            to_backend(obs), to_backend(fct_t), to_backend(alpha), **backend_kwargs
        )


def test_crps_beta(backend, to_backend, backend_kwargs):
    def fa(x):
        return to_backend(np.asarray(x, dtype=float))

    if backend == "torch":
        # betainc has no native torch implementation
        with pytest.raises(NotImplementedError):
            sr.crps_beta(
                fa(np.random.uniform(0, 1, (3, 3))),
                fa(np.random.uniform(0, 3, (3, 3))),
                fa(1.1),
            )
        return

    res = sr.crps_beta(
        fa(np.random.uniform(0, 1, (3, 3))),
        fa(np.random.uniform(0, 3, (3, 3))),
        fa(1.1),
        **backend_kwargs,
    )
    assert_inferred(res, backend)
    res_np = np.asarray(res)
    assert res_np.shape == (3, 3)
    assert not np.any(np.isnan(res_np))

    # test exceptions
    with pytest.raises(ValueError):
        sr.crps_beta(
            fa(0.3), fa(0.7), fa(1.1), lower=fa(1.0), upper=fa(0.0), **backend_kwargs
        )

    # correctness tests
    res = sr.crps_beta(fa(0.3), fa(0.7), fa(1.1), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 0.0850102437, backend)

    res = sr.crps_beta(
        fa(-3.0), fa(0.7), fa(1.1), lower=fa(-5.0), upper=fa(4.0), **backend_kwargs
    )
    assert_inferred(res, backend)
    assert_close(res, 0.883206751, backend)

    # test when lower and upper are arrays
    res = sr.crps_beta(
        fa(-3.0),
        fa(0.7),
        fa(1.1),
        lower=fa([-5.0, -5.0]),
        upper=fa([4.0, 4.0]),
        **backend_kwargs,
    )
    assert_inferred(res, backend)
    assert_close(res, np.array([0.883206751, 0.883206751]), backend)


def test_crps_binomial(backend, to_backend, backend_kwargs):
    def fa(x):
        return to_backend(np.asarray(x, dtype=float))

    if backend == "torch":
        # betainc has no native torch implementation
        with pytest.raises(NotImplementedError):
            sr.crps_binomial(fa([8, 8]), fa([10, 10]), fa([0.9, 0.9]))
        return

    # test correctness
    res = sr.crps_binomial(fa(8), fa(10), fa(0.9), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 0.6685115, backend)

    res = sr.crps_binomial(fa(-8), fa(10), fa(0.9), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 16.49896, backend)

    res = sr.crps_binomial(fa(18), fa(10), fa(0.9), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 8.498957, backend)

    # test broadcasting
    ones = np.ones(2)
    k, n, p = 8, 10, 0.9
    expected = np.array([0.6685115, 0.6685115])
    for args in [
        (fa(k * ones), fa(n), fa(p)),
        (fa(k * ones), fa(n * ones), fa(p)),
        (fa(k * ones), fa(n * ones), fa(p * ones)),
        (fa(k), fa(n * ones), fa(p * ones)),
        (fa(k * ones), fa(n), fa(p * ones)),
    ]:
        s = sr.crps_binomial(*args, **backend_kwargs)
        assert_inferred(s, backend)
        assert_close(s, expected, backend)


def test_crps_exponential(backend, to_backend, backend_kwargs):
    # TODO: add and test exception handling
    def fa(x):
        return to_backend(np.asarray(x, dtype=float))

    # test correctness
    res = sr.crps_exponential(fa(3), fa(0.7), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 1.20701837, backend)


def test_crps_exponentialM(backend, to_backend, backend_kwargs):
    def fa(x):
        return to_backend(np.asarray(x, dtype=float))

    res = sr.crps_exponentialM(fa(0.3), fa(0.1), fa(0.0), fa(1.0), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 0.2384728, backend)

    res = sr.crps_exponentialM(fa(0.3), fa(0.1), fa(-2.0), fa(3.0), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 0.6236187, backend)

    res = sr.crps_exponentialM(fa(-1.2), fa(0.1), fa(-2.0), fa(3.0), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 0.751013, backend)


def test_crps_2pexponential(backend, to_backend, backend_kwargs):
    def fa(x):
        return to_backend(np.asarray(x, dtype=float))

    res = sr.crps_2pexponential(fa(0.3), fa(0.1), fa(4.3), fa(0.0), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 1.787032, backend)

    res = sr.crps_2pexponential(
        fa(-20.8), fa(7.1), fa(2.0), fa(-25.4), **backend_kwargs
    )
    assert_inferred(res, backend)
    assert_close(res, 6.018359, backend)


def test_crps_gamma(backend, to_backend, backend_kwargs):
    def fa(x):
        return to_backend(np.asarray(x, dtype=float))

    obs, shape, rate = 0.2, 1.1, 0.7
    expected = 0.6343718

    res = sr.crps_gamma(fa(obs), fa(shape), fa(rate), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, expected, backend)

    res = sr.crps_gamma(fa(obs), fa(shape), scale=fa(1 / rate), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, expected, backend)

    with pytest.raises(ValueError):
        sr.crps_gamma(
            fa(obs), fa(shape), fa(rate), scale=fa(1 / rate), **backend_kwargs
        )

    with pytest.raises(ValueError):
        sr.crps_gamma(fa(obs), fa(shape), **backend_kwargs)


def test_crps_csg0(backend, to_backend, backend_kwargs):
    def fa(x):
        return to_backend(np.asarray(x, dtype=float))

    obs, shape, rate, shift = 0.7, 0.5, 2.0, 0.3
    expected = 0.5411044

    expected_gamma = sr.crps_gamma(fa(obs), fa(shape), fa(rate), **backend_kwargs)
    res_gamma = sr.crps_csg0(
        fa(obs), shape=fa(shape), rate=fa(rate), shift=fa(0.0), **backend_kwargs
    )
    assert_inferred(res_gamma, backend)
    assert_close(res_gamma, np.asarray(expected_gamma), backend)

    res = sr.crps_csg0(
        fa(obs), shape=fa(shape), rate=fa(rate), shift=fa(shift), **backend_kwargs
    )
    assert_inferred(res, backend)
    assert_close(res, expected, backend)

    res = sr.crps_csg0(
        fa(obs),
        shape=fa(shape),
        scale=fa(1.0 / rate),
        shift=fa(shift),
        **backend_kwargs,
    )
    assert_inferred(res, backend)
    assert_close(res, expected, backend)

    with pytest.raises(ValueError):
        sr.crps_csg0(
            fa(obs),
            shape=fa(shape),
            rate=fa(rate),
            scale=fa(1.0 / rate),
            shift=fa(shift),
            **backend_kwargs,
        )

    with pytest.raises(ValueError):
        sr.crps_csg0(fa(obs), shape=fa(shape), shift=fa(shift), **backend_kwargs)


def test_crps_gev(backend, to_backend, backend_kwargs):
    def fa(x):
        return to_backend(np.full(2, float(x)))

    if backend == "torch":
        # `expi` has no native torch implementation
        with pytest.raises(NotImplementedError):
            sr.crps_gev(fa(0.3), fa(0.0))
        return

    obs, xi = 0.3, 0.0
    res = sr.crps_gev(fa(obs), fa(xi), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 0.276440963, backend)
    res = sr.crps_gev(fa(obs + 0.1), fa(xi), location=fa(0.1), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 0.276440963, backend)
    res = sr.crps_gev(fa(obs * 0.9), fa(xi), scale=fa(0.9), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 0.276440963 * 0.9, backend)

    obs, xi = 0.3, 0.7
    res = sr.crps_gev(fa(obs), fa(xi), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 0.458044365, backend)
    res = sr.crps_gev(fa(obs + 0.1), fa(xi), location=fa(0.1), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 0.458044365, backend)
    res = sr.crps_gev(fa(obs * 0.9), fa(xi), scale=fa(0.9), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 0.458044365 * 0.9, backend)

    obs, xi = 0.3, -0.7
    res = sr.crps_gev(fa(obs), fa(xi), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 0.207621488, backend)
    res = sr.crps_gev(fa(obs + 0.1), fa(xi), location=fa(0.1), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 0.207621488, backend)
    res = sr.crps_gev(fa(obs * 0.9), fa(xi), scale=fa(0.9), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 0.207621488 * 0.9, backend)


def test_crps_gpd(backend, to_backend, backend_kwargs):
    def fa(x):
        return to_backend(np.asarray(x, dtype=float))

    res = sr.crps_gpd(fa(0.3), fa(0.9), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 0.6849332, backend)
    res = sr.crps_gpd(fa(-0.3), fa(0.9), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 1.209091, backend)
    res = sr.crps_gpd(fa(0.3), fa(-0.9), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 0.1338672, backend)
    res = sr.crps_gpd(fa(-0.3), fa(-0.9), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 0.6448276, backend)

    res = sr.crps_gpd(fa(0.3), fa(1.0), **backend_kwargs)
    assert_inferred(res, backend)
    assert np.isnan(np.asarray(res))
    res = sr.crps_gpd(fa(0.3), fa(1.2), **backend_kwargs)
    assert_inferred(res, backend)
    assert np.isnan(np.asarray(res))
    res = sr.crps_gpd(fa(0.3), fa(0.9), mass=fa(-0.1), **backend_kwargs)
    assert_inferred(res, backend)
    assert np.isnan(np.asarray(res))
    res = sr.crps_gpd(fa(0.3), fa(0.9), mass=fa(1.1), **backend_kwargs)
    assert_inferred(res, backend)
    assert np.isnan(np.asarray(res))

    expected = 0.281636441
    res = sr.crps_gpd(fa(0.3 + 0.1), fa(0.0), location=fa(0.1), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, expected, backend)
    res = sr.crps_gpd(fa(0.3 * 0.9), fa(0.0), scale=fa(0.9), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, expected * 0.9, backend)


def test_crps_gtclogis(backend, to_backend, backend_kwargs):
    def fa(x):
        return to_backend(np.asarray(x, dtype=float))

    obs, location, scale, lower, upper, lmass, umass = (
        1.8,
        -3.0,
        3.3,
        -5.0,
        4.7,
        0.1,
        0.15,
    )
    expected = 1.599721
    res = sr.crps_gtclogistic(
        fa(obs),
        fa(location),
        fa(scale),
        fa(lower),
        fa(upper),
        fa(lmass),
        fa(umass),
        **backend_kwargs,
    )
    assert_inferred(res, backend)
    assert_close(res, expected, backend)

    # aligns with crps_logistic
    res0 = sr.crps_logistic(fa(obs), fa(location), fa(scale), **backend_kwargs)
    res = sr.crps_gtclogistic(fa(obs), fa(location), fa(scale), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, np.asarray(res0), backend)

    # aligns with crps_tlogistic
    res0 = sr.crps_tlogistic(
        fa(obs), fa(location), fa(scale), fa(lower), fa(upper), **backend_kwargs
    )
    res = sr.crps_gtclogistic(
        fa(obs), fa(location), fa(scale), fa(lower), fa(upper), **backend_kwargs
    )
    assert_inferred(res, backend)
    assert_close(res, np.asarray(res0), backend)


def test_crps_tlogis(backend, to_backend, backend_kwargs):
    def fa(x):
        return to_backend(np.asarray(x, dtype=float))

    obs, location, scale, lower, upper = 4.9, 3.5, 2.3, 0.0, 20.0
    expected = 0.7658979
    res = sr.crps_tlogistic(
        fa(obs), fa(location), fa(scale), fa(lower), fa(upper), **backend_kwargs
    )
    assert_inferred(res, backend)
    assert_close(res, expected, backend)

    # aligns with crps_logistic
    res0 = sr.crps_logistic(fa(obs), fa(location), fa(scale), **backend_kwargs)
    res = sr.crps_tlogistic(fa(obs), fa(location), fa(scale), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, np.asarray(res0), backend)


def test_crps_clogis(backend, to_backend, backend_kwargs):
    def fa(x):
        return to_backend(np.asarray(x, dtype=float))

    obs, location, scale, lower, upper = -0.9, 0.4, 1.1, 0.0, 1.0
    expected = 1.13237
    res = sr.crps_clogistic(
        fa(obs), fa(location), fa(scale), fa(lower), fa(upper), **backend_kwargs
    )
    assert_inferred(res, backend)
    assert_close(res, expected, backend)

    # aligns with crps_logistic
    res0 = sr.crps_logistic(fa(obs), fa(location), fa(scale), **backend_kwargs)
    res = sr.crps_clogistic(fa(obs), fa(location), fa(scale), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, np.asarray(res0), backend)


def test_crps_gtcnormal(backend, to_backend, backend_kwargs):
    def fa(x):
        return to_backend(np.asarray(x, dtype=float))

    obs, location, scale, lower, upper, lmass, umass = (
        0.9,
        -2.3,
        4.1,
        -7.3,
        1.7,
        0.0,
        0.21,
    )
    expected = 1.422805
    res = sr.crps_gtcnormal(
        fa(obs),
        fa(location),
        fa(scale),
        fa(lower),
        fa(upper),
        fa(lmass),
        fa(umass),
        **backend_kwargs,
    )
    assert_inferred(res, backend)
    assert_close(res, expected, backend)

    # aligns with crps_normal
    res0 = sr.crps_normal(fa(obs), fa(location), fa(scale), **backend_kwargs)
    res = sr.crps_gtcnormal(fa(obs), fa(location), fa(scale), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, np.asarray(res0), backend)

    # aligns with crps_tnormal
    res0 = sr.crps_tnormal(
        fa(obs), fa(location), fa(scale), fa(lower), fa(upper), **backend_kwargs
    )
    res = sr.crps_gtcnormal(
        fa(obs), fa(location), fa(scale), fa(lower), fa(upper), **backend_kwargs
    )
    assert_inferred(res, backend)
    assert_close(res, np.asarray(res0), backend)


def test_crps_tnormal(backend, to_backend, backend_kwargs):
    def fa(x):
        return to_backend(np.asarray(x, dtype=float))

    obs, location, scale, lower, upper = -1.0, 2.9, 2.2, 1.5, 17.3
    expected = 3.982434
    res = sr.crps_tnormal(
        fa(obs), fa(location), fa(scale), fa(lower), fa(upper), **backend_kwargs
    )
    assert_inferred(res, backend)
    assert_close(res, expected, backend)

    # aligns with crps_normal
    res0 = sr.crps_normal(fa(obs), fa(location), fa(scale), **backend_kwargs)
    res = sr.crps_tnormal(fa(obs), fa(location), fa(scale), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, np.asarray(res0), backend)


def test_crps_cnormal(backend, to_backend, backend_kwargs):
    def fa(x):
        return to_backend(np.asarray(x, dtype=float))

    obs, location, scale, lower, upper = 1.8, 0.4, 1.1, 0.0, 2.0
    expected = 0.8296078
    res = sr.crps_cnormal(
        fa(obs), fa(location), fa(scale), fa(lower), fa(upper), **backend_kwargs
    )
    assert_inferred(res, backend)
    assert_close(res, expected, backend)

    # aligns with crps_normal (default unbounded). The infinite-bound arithmetic
    # overflows to NaN in jax's float32; it is correct under float64 (jax x64), so
    # only the numeric comparison is skipped on jax -- inference is still asserted.
    res0 = sr.crps_normal(fa(obs), fa(location), fa(scale), **backend_kwargs)
    res = sr.crps_cnormal(fa(obs), fa(location), fa(scale), **backend_kwargs)
    assert_inferred(res, backend)
    if backend != "jax":
        assert_close(res, np.asarray(res0), backend)


def test_crps_gtct(backend, to_backend, backend_kwargs):
    def fa(x):
        return to_backend(np.asarray(x, dtype=float))

    if backend == "torch":
        # betainc/hyp2f1 have no native torch implementation
        with pytest.raises(NotImplementedError):
            sr.crps_gtct(
                fa(0.9),
                fa(20.1),
                fa(-2.3),
                fa(4.1),
                fa(-7.3),
                fa(1.7),
                fa(0.0),
                fa(0.21),
            )
        return

    obs, df, location, scale, lower, upper, lmass, umass = (
        0.9,
        20.1,
        -2.3,
        4.1,
        -7.3,
        1.7,
        0.0,
        0.21,
    )
    expected = 1.423042
    res = sr.crps_gtct(
        fa(obs),
        fa(df),
        fa(location),
        fa(scale),
        fa(lower),
        fa(upper),
        fa(lmass),
        fa(umass),
        **backend_kwargs,
    )
    assert_inferred(res, backend)
    assert_close(res, expected, backend)

    # aligns with crps_t
    res0 = sr.crps_t(fa(obs), fa(df), fa(location), fa(scale), **backend_kwargs)
    res = sr.crps_gtct(fa(obs), fa(df), fa(location), fa(scale), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, np.asarray(res0), backend)

    # aligns with crps_tt
    res0 = sr.crps_tt(
        fa(obs), fa(df), fa(location), fa(scale), fa(lower), fa(upper), **backend_kwargs
    )
    res = sr.crps_gtct(
        fa(obs), fa(df), fa(location), fa(scale), fa(lower), fa(upper), **backend_kwargs
    )
    assert_inferred(res, backend)
    assert_close(res, np.asarray(res0), backend)


def test_crps_tt(backend, to_backend, backend_kwargs):
    def fa(x):
        return to_backend(np.asarray(x, dtype=float))

    if backend == "torch":
        # betainc/hyp2f1 have no native torch implementation
        with pytest.raises(NotImplementedError):
            sr.crps_tt(fa(-1.0), fa(2.9), fa(3.1), fa(4.2), fa(1.5), fa(17.3))
        return

    obs, df, location, scale, lower, upper = -1.0, 2.9, 3.1, 4.2, 1.5, 17.3
    expected = 5.084272
    res = sr.crps_tt(
        fa(obs), fa(df), fa(location), fa(scale), fa(lower), fa(upper), **backend_kwargs
    )
    assert_inferred(res, backend)
    assert_close(res, expected, backend)

    # aligns with crps_t
    res0 = sr.crps_t(fa(obs), fa(df), fa(location), fa(scale), **backend_kwargs)
    res = sr.crps_tt(fa(obs), fa(df), fa(location), fa(scale), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, np.asarray(res0), backend)


def test_crps_ct(backend, to_backend, backend_kwargs):
    def fa(x):
        return to_backend(np.asarray(x, dtype=float))

    if backend == "torch":
        # betainc/hyp2f1 have no native torch implementation
        with pytest.raises(NotImplementedError):
            sr.crps_ct(fa(1.8), fa(5.4), fa(0.4), fa(1.1), fa(0.0), fa(2.0))
        return

    obs, df, location, scale, lower, upper = 1.8, 5.4, 0.4, 1.1, 0.0, 2.0
    expected = 0.8028996
    res = sr.crps_ct(
        fa(obs), fa(df), fa(location), fa(scale), fa(lower), fa(upper), **backend_kwargs
    )
    assert_inferred(res, backend)
    assert_close(res, expected, backend)

    # aligns with crps_t
    res0 = sr.crps_t(fa(obs), fa(df), fa(location), fa(scale), **backend_kwargs)
    res = sr.crps_ct(fa(obs), fa(df), fa(location), fa(scale), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, np.asarray(res0), backend)


def test_crps_hypergeometric(backend, to_backend, backend_kwargs):
    def fa(x):
        return to_backend(np.asarray(x, dtype=float))

    # test shapes
    res = sr.crps_hypergeometric(
        fa(5 * np.ones((2, 2))), fa(7), fa(13), fa(12), **backend_kwargs
    )
    assert_inferred(res, backend)
    assert np.asarray(res).shape == (2, 2)

    res = sr.crps_hypergeometric(
        fa(5), fa(7 * np.ones((2, 2))), fa(13), fa(12), **backend_kwargs
    )
    assert_inferred(res, backend)
    assert np.asarray(res).shape == (2, 2)

    res = sr.crps_hypergeometric(
        fa(5), fa(7), fa(13 * np.ones((2, 2))), fa(12), **backend_kwargs
    )
    assert_inferred(res, backend)
    assert np.asarray(res).shape == (2, 2)

    # test correctness
    res = sr.crps_hypergeometric(fa(5), fa(7), fa(13), fa(12), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 0.4469742, backend)


def test_crps_laplace(backend, to_backend, backend_kwargs):
    def fa(x):
        return to_backend(np.asarray(x, dtype=float))

    res = sr.crps_laplace(fa(-3), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 2.29978707, backend)

    res = sr.crps_laplace(fa(-3 + 0.1), location=fa(0.1), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 2.29978707, backend)

    res = sr.crps_laplace(fa(-3 * 0.9), scale=fa(0.9), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 0.9 * 2.29978707, backend)


def test_crps_logis(backend, to_backend, backend_kwargs):
    def fa(x):
        return to_backend(np.asarray(x, dtype=float))

    res = sr.crps_logistic(fa(17.1), fa(13.8), fa(3.3), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 2.067527, backend)

    res = sr.crps_logistic(fa(3.1), fa(4.0), fa(0.5), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 0.5529776, backend)


def test_crps_loglaplace(backend, to_backend, backend_kwargs):
    def fa(x):
        return to_backend(np.asarray(x, dtype=float))

    res = sr.crps_loglaplace(fa(3.0), fa(0.1), fa(0.9), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 1.16202051, backend)


def test_crps_loglogistic(backend, to_backend, backend_kwargs):
    def fa(x):
        return to_backend(np.asarray(x, dtype=float))

    if backend == "torch":
        # hyp2f1 has no native torch implementation
        with pytest.raises(NotImplementedError):
            sr.crps_loglogistic(fa(3.0), fa(0.1), fa(0.9))
        return

    # JAX results differ slightly from other backends -> looser atol
    res = sr.crps_loglogistic(fa(3.0), fa(0.1), fa(0.9), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 1.13295277, backend, atol=1e-4)


def test_crps_lognormal(backend, to_backend, backend_kwargs):
    obs0 = np.exp(np.random.randn(N))
    mulog = np.log(obs0) + np.random.randn(N) * 0.1
    sigmalog = abs(np.random.randn(N)) * 0.3

    # non-negative values
    res = sr.crps_lognormal(
        to_backend(obs0), to_backend(mulog), to_backend(sigmalog), **backend_kwargs
    )
    assert_inferred(res, backend)
    res = np.asarray(res)
    assert not np.any(np.isnan(res))
    assert not np.any(res < 0.0)

    # approx zero when perfect forecast
    mulog = np.log(obs0) + np.random.randn(N) * 1e-6
    sigmalog = abs(np.random.randn(N)) * 1e-6
    res = sr.crps_lognormal(
        to_backend(obs0), to_backend(mulog), to_backend(sigmalog), **backend_kwargs
    )
    assert_inferred(res, backend)
    res = np.asarray(res)
    assert not np.any(np.isnan(res))
    assert not np.any(res - 0.0 > 0.0001)


def test_crps_mixnorm(backend, to_backend, backend_kwargs):
    def fa(x):
        return to_backend(np.asarray(x, dtype=float))

    obs, m, s, w = 0.3, [0.0, -2.9, 0.9], [0.5, 1.4, 0.7], [1 / 3, 1 / 3, 1 / 3]
    res = sr.crps_mixnorm(fa(obs), fa(m), fa(s), fa(w), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 0.4510451, backend)

    res0 = sr.crps_mixnorm(fa(obs), fa(m), fa(s), **backend_kwargs)
    assert_inferred(res0, backend)
    assert_close(res, np.asarray(res0), backend)

    w = [0.3, 0.1, 0.6]
    res = sr.crps_mixnorm(fa(obs), fa(m), fa(s), fa(w), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 0.2354619, backend)

    obs = [-1.6, 0.3]
    m = [[0.0, -2.9], [0.6, 0.0], [-1.1, -2.3]]
    s = [[0.5, 1.7], [1.1, 0.7], [1.4, 1.5]]
    res1 = sr.crps_mixnorm(fa(obs), fa(m), fa(s), m_axis=0, **backend_kwargs)
    assert_inferred(res1, backend)

    m = [[0.0, 0.6, -1.1], [-2.9, 0.0, -2.3]]
    s = [[0.5, 1.1, 1.4], [1.7, 0.7, 1.5]]
    res2 = sr.crps_mixnorm(fa(obs), fa(m), fa(s), **backend_kwargs)
    assert_inferred(res2, backend)
    assert_close(res1, np.asarray(res2), backend)


def test_crps_negbinom(backend, to_backend, backend_kwargs):
    def fa(x):
        return to_backend(np.asarray(x, dtype=float))

    if backend == "torch":
        # betainc/hyp2f1 have no native torch implementation
        with pytest.raises(NotImplementedError):
            sr.crps_negbinom(fa(2.0), fa(7.0), fa(0.8))
        return

    # test exceptions
    with pytest.raises(ValueError):
        sr.crps_negbinom(fa(0.3), fa(7.0), fa(0.8), mu=fa(7.3), **backend_kwargs)

    # test correctness
    res = sr.crps_negbinom(fa(2.0), fa(7.0), fa(0.8), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 0.3834322, backend)

    res = sr.crps_negbinom(fa(1.5), fa(2.0), fa(0.5), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 0.462963, backend)

    # large-magnitude case: overflows to NaN in jax's float32 but is correct under
    # float64 (jax x64), so only the numeric comparison is skipped on jax.
    res = sr.crps_negbinom(fa(-1.0), fa(17.0), fa(0.1), **backend_kwargs)
    assert_inferred(res, backend)
    if backend != "jax":
        assert_close(res, 132.0942, backend)

    res = sr.crps_negbinom(fa(2.3), fa(11.0), mu=fa(7.3), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 3.149218, backend)


def test_crps_normal(backend, to_backend, backend_kwargs):
    obs0 = np.random.randn(N)
    mu = obs0 + np.random.randn(N) * 0.1
    sigma = abs(np.random.randn(N)) * 0.3

    # non-negative values
    res = sr.crps_normal(
        to_backend(obs0), to_backend(mu), to_backend(sigma), **backend_kwargs
    )
    assert_inferred(res, backend)
    res = np.asarray(res)
    assert not np.any(np.isnan(res))
    assert not np.any(res < 0.0)

    # approx zero when perfect forecast
    mu = obs0 + np.random.randn(N) * 1e-6
    sigma = abs(np.random.randn(N)) * 1e-6
    res = sr.crps_normal(
        to_backend(obs0), to_backend(mu), to_backend(sigma), **backend_kwargs
    )
    assert_inferred(res, backend)
    res = np.asarray(res)
    assert not np.any(np.isnan(res))
    assert not np.any(res - 0.0 > 0.0001)


def test_crps_2pnormal(backend, to_backend, backend_kwargs):
    obs0 = np.random.randn(N)
    mu = obs0 + np.random.randn(N) * 0.1
    sigma1 = abs(np.random.randn(N)) * 0.3
    sigma2 = abs(np.random.randn(N)) * 0.2

    res = sr.crps_2pnormal(
        to_backend(obs0),
        to_backend(sigma1),
        to_backend(sigma2),
        to_backend(mu),
        **backend_kwargs,
    )
    assert_inferred(res, backend)
    res = np.asarray(res)
    assert not np.any(np.isnan(res))
    assert not np.any(res < 0.0)


def test_crps_poisson(backend, to_backend, backend_kwargs):
    def fa(x):
        return to_backend(np.asarray(x, dtype=float))

    res = sr.crps_poisson(fa(1.0), fa(3.0), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 1.143447, backend)

    res = sr.crps_poisson(fa(1.5), fa(2.3), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 0.5001159, backend)

    res = sr.crps_poisson(fa(-1.0), fa(1.5), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 1.840259, backend)


def test_crps_t(backend, to_backend, backend_kwargs):
    def fa(x):
        return to_backend(np.asarray(x, dtype=float))

    if backend == "torch":
        # betainc has no native torch implementation
        with pytest.raises(NotImplementedError):
            sr.crps_t(fa(11.1), fa(5.2), fa(13.8), fa(2.3))
        return

    res = sr.crps_t(fa(11.1), fa(5.2), fa(13.8), fa(2.3), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 1.658226, backend)

    res = sr.crps_t(fa(0.7), fa(4.0), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 0.4387929, backend)


def test_crps_uniform(backend, to_backend, backend_kwargs):
    def fa(x):
        return to_backend(np.asarray(x, dtype=float))

    res = sr.crps_uniform(
        fa(0.3), fa(-1.0), fa(2.1), fa(0.3), fa(0.1), **backend_kwargs
    )
    assert_inferred(res, backend)
    assert_close(res, 0.3960968, backend)

    res = sr.crps_uniform(fa(-17.9), fa(-15.2), fa(-8.7), fa(0.2), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 4.086667, backend)

    res = sr.crps_uniform(fa(2.2), fa(0.1), fa(3.1), **backend_kwargs)
    assert_inferred(res, backend)
    assert_close(res, 0.37, backend)


def test_crps_ensemble_legacy_backend_string_deprecated():
    obs = np.random.randn(N)
    fct = np.random.randn(N, ENSEMBLE_SIZE)
    with pytest.warns(DeprecationWarning):
        sr.crps_ensemble(obs, fct, backend="numpy")


def test_crps_ensemble_grad_jax():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    obs = jnp.asarray(np.random.randn(N))
    fct = jnp.asarray(np.random.randn(N, ENSEMBLE_SIZE))
    g = jax.grad(lambda f: sr.crps_ensemble(obs, f).sum())(fct)
    assert jnp.all(jnp.isfinite(g))


def test_crps_normal_grad_torch():
    torch = pytest.importorskip("torch")
    obs = torch.tensor([0.2], dtype=torch.float64)
    mu = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
    sigma = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    sr.crps_normal(obs, mu, sigma).sum().backward()
    assert torch.isfinite(mu.grad).all() and torch.isfinite(sigma.grad).all()
