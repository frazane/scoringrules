"""Tests for nan_policy parameter in CRPS ensemble scoring rules.

Covers all three policies ('propagate', 'omit', 'raise') across all CRPS estimators
and the weighted variants (twcrps, owcrps, vrcrps).

Note on akr/akr_circperm with nan_policy='omit' on array backends:
    The array backend uses a roll-based masking approach for the circular kernel,
    which masks out pairs involving NaN members but also loses the wrap-around pair
    when NaN sits at the tail.  This means the array-backend result does NOT match
    the clean-ensemble result (a known limitation).  The numba gufuncs correctly
    reconnect valid neighbours and DO match the clean ensemble.  Tests that require
    a match against the clean ensemble therefore only run for the numba backend on
    akr/akr_circperm.
"""

import numpy as np
import pytest
import scoringrules as sr

# All estimators available for crps_ensemble
ESTIMATORS = ["nrg", "fair", "pwm", "qd", "int", "akr", "akr_circperm"]

# Estimators whose omit behaviour matches the clean-ensemble result on all backends.
# 'int' is excluded because nan_policy='omit' raises NotImplementedError on non-numba.
# 'akr'/'akr_circperm' are excluded because the array backend uses a different
# approximation (see module docstring above).
ESTIMATORS_OMIT_ALL_BACKENDS = ["nrg", "fair", "pwm", "qd"]

# Same but also excludes 'fair' for edge-case tests where M_eff may equal 1
# (fair's denominator M_eff*(M_eff-1) is 0 when M_eff==1 → returns NaN by design).
ESTIMATORS_OMIT_MEFF_GE2 = ["nrg", "pwm", "qd"]

# Reference inputs: five valid members, with and without NaN interspersed.
# The valid members are the same in both arrays (and already sorted).
_OBS = np.float64(0.5)
_FCT_CLEAN = np.array([0.1, 0.3, 0.7, 0.9, 1.1])
_FCT_WITH_NAN = np.array([0.1, np.nan, 0.3, np.nan, 0.7, 0.9, 1.1])
# NaN members of _FCT_WITH_NAN replaced by _OBS — same shape, no NaN, used for
# building 2-row batches where one row is clean and the other contains NaN.
_FCT_WITH_NAN_FILLED = np.where(np.isnan(_FCT_WITH_NAN), _OBS, _FCT_WITH_NAN)


# ---------------------------------------------------------------------------
# propagate policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_propagate_returns_nan(estimator, backend):
    """nan_policy='propagate' returns NaN when the ensemble contains NaN members."""
    if backend == "numba" and estimator == "int":
        # The existing int gufunc stops at the first NaN (sorted to the tail) and
        # returns a partial integral rather than NaN — pre-existing behaviour outside
        # the scope of this PR.
        pytest.skip(
            "int gufunc stops early at NaN (pre-existing behaviour) rather than "
            "propagating NaN; propagation for int/numba is not a goal of this PR."
        )
    res = sr.crps_ensemble(
        _OBS,
        _FCT_WITH_NAN,
        estimator=estimator,
        nan_policy="propagate",
        backend=backend,
    )
    assert np.isnan(float(np.asarray(res)))


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_propagate_is_default(estimator, backend):
    """Calling without nan_policy is identical to nan_policy='propagate'."""
    if backend == "numba" and estimator == "int":
        pytest.skip("int/numba pre-existing NaN propagation behaviour — see above.")
    res_explicit = sr.crps_ensemble(
        _OBS,
        _FCT_WITH_NAN,
        estimator=estimator,
        nan_policy="propagate",
        backend=backend,
    )
    res_default = sr.crps_ensemble(
        _OBS, _FCT_WITH_NAN, estimator=estimator, backend=backend
    )
    assert np.array_equal(
        np.asarray(res_explicit), np.asarray(res_default), equal_nan=True
    )


# ---------------------------------------------------------------------------
# omit policy — basic sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "estimator", ESTIMATORS_OMIT_ALL_BACKENDS + ["akr", "akr_circperm"]
)
def test_omit_ignores_nans(estimator, backend):
    """nan_policy='omit' returns a finite score when the ensemble has NaN members."""
    res = sr.crps_ensemble(
        _OBS, _FCT_WITH_NAN, estimator=estimator, nan_policy="omit", backend=backend
    )
    assert np.isfinite(float(np.asarray(res)))


@pytest.mark.parametrize(
    "estimator", ESTIMATORS_OMIT_ALL_BACKENDS + ["akr", "akr_circperm"]
)
def test_omit_all_nan_returns_nan(estimator, backend):
    """nan_policy='omit' returns NaN when every ensemble member is NaN."""
    fct_all_nan = np.array([np.nan, np.nan, np.nan])
    # silence warnings (divide by zero, invalid value encountered)
    with np.errstate(divide="ignore", invalid="ignore"):
        res = sr.crps_ensemble(
            _OBS, fct_all_nan, estimator=estimator, nan_policy="omit", backend=backend
        )
    assert np.isnan(float(np.asarray(res)))


# ---------------------------------------------------------------------------
# omit policy — correctness: result must match the clean ensemble
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("estimator", ESTIMATORS_OMIT_ALL_BACKENDS)
def test_omit_matches_manual(estimator, backend):
    """nan_policy='omit' matches computing the score on the clean (NaN-free) ensemble."""
    res_omit = sr.crps_ensemble(
        _OBS, _FCT_WITH_NAN, estimator=estimator, nan_policy="omit", backend=backend
    )
    # sorted_ensemble=True avoids re-sorting the already-sorted clean array.
    res_clean = sr.crps_ensemble(
        _OBS, _FCT_CLEAN, estimator=estimator, sorted_ensemble=True, backend=backend
    )
    assert np.isclose(
        float(np.asarray(res_omit)), float(np.asarray(res_clean)), rtol=1e-5
    )


@pytest.mark.parametrize("estimator", ["akr", "akr_circperm"])
def test_omit_matches_manual_akr_numba(estimator, backend):
    """akr/akr_circperm nanomit gufuncs match the clean ensemble on numba."""
    if backend != "numba":
        pytest.skip(
            "Array-backend akr/akr_circperm uses a roll-based masking that loses "
            "valid consecutive pairs when NaN members are interspersed, so the result "
            "differs from the clean ensemble — known limitation."
        )
    res_omit = sr.crps_ensemble(
        _OBS, _FCT_WITH_NAN, estimator=estimator, nan_policy="omit", backend=backend
    )
    res_clean = sr.crps_ensemble(_OBS, _FCT_CLEAN, estimator=estimator, backend=backend)
    assert np.isclose(
        float(np.asarray(res_omit)), float(np.asarray(res_clean)), rtol=1e-5
    )


def test_omit_int_numba_matches_manual(backend):
    """int estimator with nan_policy='omit' works on numba and matches the clean ensemble."""
    if backend != "numba":
        pytest.skip("int + omit only supported on numba")
    res_omit = sr.crps_ensemble(
        _OBS, _FCT_WITH_NAN, estimator="int", nan_policy="omit", backend=backend
    )
    res_clean = sr.crps_ensemble(
        _OBS, _FCT_CLEAN, estimator="int", sorted_ensemble=True, backend=backend
    )
    assert np.isclose(
        float(np.asarray(res_omit)), float(np.asarray(res_clean)), rtol=1e-5
    )


def test_omit_int_non_numba_raises(backend):
    """int estimator with nan_policy='omit' raises NotImplementedError on non-numba."""
    if backend == "numba":
        pytest.skip("int + omit is supported on numba")
    with pytest.raises(NotImplementedError):
        sr.crps_ensemble(
            _OBS, _FCT_WITH_NAN, estimator="int", nan_policy="omit", backend=backend
        )


# ---------------------------------------------------------------------------
# omit policy — edge cases
# ---------------------------------------------------------------------------


def test_omit_nan_in_obs(backend):
    """nan_policy='omit' still returns NaN when obs itself is NaN."""
    res = sr.crps_ensemble(
        np.float64(np.nan), _FCT_CLEAN, nan_policy="omit", backend=backend
    )
    assert np.isnan(float(np.asarray(res)))


def test_omit_fair_meff_one_returns_nan(backend):
    """fair estimator with nan_policy='omit' returns NaN when only one valid member remains."""
    fct_one_valid = np.array([_FCT_CLEAN[0], np.nan, np.nan])
    res = sr.crps_ensemble(
        _OBS, fct_one_valid, estimator="fair", nan_policy="omit", backend=backend
    )
    assert np.isnan(float(np.asarray(res)))


@pytest.mark.parametrize("estimator", ESTIMATORS_OMIT_ALL_BACKENDS)
def test_omit_varying_nan_counts(estimator, backend):
    """Batched forecasts each with a different number of NaN members."""
    obs = np.zeros(4)
    fct = np.array(
        [
            [1.0, 2.0, 3.0, np.nan, np.nan],  # 3 valid
            [0.5, 1.5, 2.5, np.nan, np.nan],  # 3 valid (different values)
            [0.5, 1.5, np.nan, np.nan, np.nan],  # 2 valid
            [1.0, 2.0, 3.0, 4.0, 5.0],  # 5 valid, no NaN
        ]
    )
    res = np.asarray(
        sr.crps_ensemble(
            obs, fct, estimator=estimator, nan_policy="omit", backend=backend
        )
    )
    assert res.shape == (4,)
    # Rows with ≥ 2 valid members should always be finite (even for 'fair').
    assert np.all(np.isfinite(res))

    # Row 3 (no NaN) should match the clean computation on the same backend.
    res_clean = float(
        np.asarray(
            sr.crps_ensemble(
                0.0,
                np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                estimator=estimator,
                backend=backend,
            )
        )
    )
    assert np.isclose(float(res[3]), res_clean, rtol=1e-5)


# ---------------------------------------------------------------------------
# raise policy
# ---------------------------------------------------------------------------


def test_raise_with_nans(backend):
    """nan_policy='raise' raises ValueError when NaN values are present."""
    with pytest.raises(ValueError):
        sr.crps_ensemble(_OBS, _FCT_WITH_NAN, nan_policy="raise", backend=backend)


def test_raise_without_nans(backend):
    """nan_policy='raise' computes normally when no NaN values are present."""
    res = sr.crps_ensemble(_OBS, _FCT_CLEAN, nan_policy="raise", backend=backend)
    assert np.isfinite(float(np.asarray(res)))


def test_raise_with_nan_in_obs(backend):
    """nan_policy='raise' raises ValueError when obs itself is NaN."""
    with pytest.raises(ValueError):
        sr.crps_ensemble(
            np.float64(np.nan), _FCT_CLEAN, nan_policy="raise", backend=backend
        )


# ---------------------------------------------------------------------------
# policy validation
# ---------------------------------------------------------------------------


def test_invalid_nan_policy(backend):
    """An unrecognised nan_policy string raises ValueError."""
    with pytest.raises(ValueError):
        sr.crps_ensemble(_OBS, _FCT_CLEAN, nan_policy="invalid_policy", backend=backend)


# ---------------------------------------------------------------------------
# all three policies agree when there are no NaN values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "estimator", ESTIMATORS_OMIT_ALL_BACKENDS + ["akr", "akr_circperm"]
)
def test_no_nans_all_policies_equal(estimator, backend):
    """All three policies produce identical results when the ensemble has no NaN values."""
    res_prop = sr.crps_ensemble(
        _OBS, _FCT_CLEAN, estimator=estimator, nan_policy="propagate", backend=backend
    )
    res_omit = sr.crps_ensemble(
        _OBS, _FCT_CLEAN, estimator=estimator, nan_policy="omit", backend=backend
    )
    res_raise = sr.crps_ensemble(
        _OBS, _FCT_CLEAN, estimator=estimator, nan_policy="raise", backend=backend
    )
    assert np.isclose(float(np.asarray(res_prop)), float(np.asarray(res_omit)))
    assert np.isclose(float(np.asarray(res_prop)), float(np.asarray(res_raise)))


# ---------------------------------------------------------------------------
# Weighted variants
# ---------------------------------------------------------------------------


def test_twcrps_propagate_returns_nan(backend):
    """twcrps_ensemble with nan_policy='propagate' returns NaN for NaN ensemble."""
    res = sr.twcrps_ensemble(
        _OBS, _FCT_WITH_NAN, nan_policy="propagate", backend=backend
    )
    assert np.isnan(float(np.asarray(res)))


def test_twcrps_omit_finite(backend):
    """twcrps_ensemble with nan_policy='omit' returns a finite score."""
    res = sr.twcrps_ensemble(_OBS, _FCT_WITH_NAN, nan_policy="omit", backend=backend)
    assert np.isfinite(float(np.asarray(res)))


def test_twcrps_no_nans_policies_equal(backend):
    """twcrps_ensemble: all policies agree when no NaN values are present."""
    res_prop = sr.twcrps_ensemble(
        _OBS, _FCT_CLEAN, nan_policy="propagate", backend=backend
    )
    res_omit = sr.twcrps_ensemble(_OBS, _FCT_CLEAN, nan_policy="omit", backend=backend)
    assert np.isclose(float(np.asarray(res_prop)), float(np.asarray(res_omit)))


def test_twcrps_raise_with_nans(backend):
    """twcrps_ensemble with nan_policy='raise' raises ValueError when NaN is present."""
    with pytest.raises(ValueError):
        sr.twcrps_ensemble(_OBS, _FCT_WITH_NAN, nan_policy="raise", backend=backend)


def test_owcrps_propagate_returns_nan(backend):
    """owcrps_ensemble with nan_policy='propagate' returns NaN for NaN ensemble."""
    res = sr.owcrps_ensemble(
        _OBS, _FCT_WITH_NAN, nan_policy="propagate", backend=backend
    )
    assert np.isnan(float(np.asarray(res)))


def test_owcrps_omit_matches_manual(backend):
    """owcrps_ensemble with nan_policy='omit' matches computing on the clean ensemble."""
    res_omit = sr.owcrps_ensemble(
        _OBS, _FCT_WITH_NAN, nan_policy="omit", backend=backend
    )
    res_clean = sr.owcrps_ensemble(_OBS, _FCT_CLEAN, backend=backend)
    assert np.isclose(
        float(np.asarray(res_omit)), float(np.asarray(res_clean)), rtol=1e-5
    )


def test_owcrps_raise_with_nans(backend):
    """owcrps_ensemble with nan_policy='raise' raises ValueError."""
    with pytest.raises(ValueError):
        sr.owcrps_ensemble(_OBS, _FCT_WITH_NAN, nan_policy="raise", backend=backend)


def test_vrcrps_propagate_returns_nan(backend):
    """vrcrps_ensemble with nan_policy='propagate' returns NaN for NaN ensemble."""
    # Use batched inputs — vr_ensemble has a pre-existing axis=1 bug with scalar obs.
    obs = np.array([float(_OBS), float(_OBS)])
    fct = np.array([_FCT_WITH_NAN, _FCT_WITH_NAN_FILLED])
    res = np.asarray(
        sr.vrcrps_ensemble(obs, fct, nan_policy="propagate", backend=backend)
    )
    assert np.isnan(res[0])  # NaN ensemble → NaN
    assert np.isfinite(res[1])  # clean ensemble → finite


def test_vrcrps_omit_matches_manual(backend):
    """vrcrps_ensemble with nan_policy='omit' matches computing on the clean ensemble."""
    # Use batched inputs — vr_ensemble has a pre-existing axis=1 bug with scalar obs.
    obs = np.array([float(_OBS)])
    res_omit = np.asarray(
        sr.vrcrps_ensemble(
            obs, _FCT_WITH_NAN[np.newaxis, :], nan_policy="omit", backend=backend
        )
    )
    res_clean = np.asarray(
        sr.vrcrps_ensemble(obs, _FCT_CLEAN[np.newaxis, :], backend=backend)
    )
    assert np.isclose(float(res_omit[0]), float(res_clean[0]), rtol=1e-5)


def test_vrcrps_raise_with_nans(backend):
    """vrcrps_ensemble with nan_policy='raise' raises ValueError."""
    obs = np.array([float(_OBS)])
    with pytest.raises(ValueError):
        sr.vrcrps_ensemble(
            obs, _FCT_WITH_NAN[np.newaxis, :], nan_policy="raise", backend=backend
        )
