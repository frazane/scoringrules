import numpy as np
import pytest
from scoringrules.backend import get_namespace
from scoringrules.core import utils_xp


def test_univariate_array_check_moves_axis():
    obs = np.random.randn(5)
    fct = np.random.randn(11, 5)  # ensemble axis first
    xp = get_namespace(obs, fct)
    o, f = utils_xp.univariate_array_check(obs, fct, m_axis=0, xp=xp)
    assert f.shape == (5, 11)


def test_nan_policy_propagate_is_noop():
    obs = np.random.randn(5)
    fct = np.random.randn(5, 11)
    xp = get_namespace(obs, fct)
    o, f, w = utils_xp.apply_nan_policy_ens_uv(obs, fct, "propagate", xp=xp)
    assert w is None


def test_xp_helpers_match_original_numpy():
    from scoringrules.core import utils as orig

    obs = np.random.randn(4)
    fct = np.random.randn(4, 7)
    xp = get_namespace(obs, fct)
    # weight normalisation parity
    w = np.abs(np.random.randn(4, 7)) + 0.1
    w_xp = utils_xp.univariate_weight_check(w.copy(), fct, -1, xp=xp)
    w_orig = orig.univariate_weight_check(w.copy(), fct, -1, backend="numpy")
    assert np.asarray(w_xp) == pytest.approx(np.asarray(w_orig))
    # nan omit parity (mask building)
    fct_nan = fct.copy()
    fct_nan[0, [1, 3]] = np.nan
    _, f_xp, m_xp = utils_xp.apply_nan_policy_ens_uv(
        obs, fct_nan.copy(), "omit", estimator="nrg", xp=xp
    )
    _, f_o, m_o = orig.apply_nan_policy_ens_uv(
        obs, fct_nan.copy(), "omit", estimator="nrg", m_axis=-1, backend="numpy"
    )
    assert np.asarray(m_xp) == pytest.approx(np.asarray(m_o))
