import numpy as np
import pytest
from scoringrules.backend import get_namespace
from scoringrules.core import utils


def test_univariate_array_check_moves_axis():
    obs = np.random.randn(5)
    fct = np.random.randn(11, 5)  # ensemble axis first
    xp = get_namespace(obs, fct)
    o, f = utils.univariate_array_check(obs, fct, m_axis=0, xp=xp)
    assert f.shape == (5, 11)


def test_nan_policy_propagate_is_noop():
    obs = np.random.randn(5)
    fct = np.random.randn(5, 11)
    xp = get_namespace(obs, fct)
    o, f, w = utils.apply_nan_policy_ens_uv(obs, fct, "propagate", xp=xp)
    assert w is None


def test_xp_helpers_match_original_numpy():
    obs = np.random.randn(4)
    fct = np.random.randn(4, 7)
    xp = get_namespace(obs, fct)
    # weight normalisation (the legacy helper is gone; compare with the formula)
    w = np.abs(np.random.randn(4, 7)) + 0.1
    w_xp = utils.univariate_weight_check(w.copy(), fct, -1, xp=xp)
    assert np.asarray(w_xp) == pytest.approx(w / w.sum(axis=-1, keepdims=True))
    # nan omit mask building (the legacy numpy helper is gone, the numba path
    # uses this one too)
    fct_nan = fct.copy()
    fct_nan[0, [1, 3]] = np.nan
    _, f_xp, m_xp = utils.apply_nan_policy_ens_uv(
        obs, fct_nan.copy(), "omit", estimator="nrg", xp=xp
    )
    np.testing.assert_array_equal(m_xp, (~np.isnan(fct_nan)).astype(float))
    np.testing.assert_array_equal(f_xp, np.where(np.isnan(fct_nan), 0.0, fct_nan))
