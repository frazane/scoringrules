import numpy as np
import pytest
import scoringrules as sr

# #148: lazy gufuncs allocated their output like ``obs``, so integer observations
# gave integer scores (or a UFuncTypeError for the multivariate ones).
_rng = np.random.default_rng(42)
_FCT_UV = _rng.normal(size=(3, 7))
_FCT_MV = _rng.normal(size=(3, 7, 2))
CASES = {
    "crps_ensemble_pwm": (
        lambda o, **kw: sr.crps_ensemble(o, _FCT_UV, estimator="pwm", **kw),
        np.array([0, 1, 2]),
    ),
    "error_spread_score": (
        lambda o, **kw: sr.error_spread_score(o, _FCT_UV, **kw),
        np.array([0, 1, 2]),
    ),
    "dssuv_ensemble": (
        lambda o, **kw: sr.dssuv_ensemble(o, _FCT_UV, **kw),
        np.array([0, 1, 2]),
    ),
    "es_ensemble": (
        lambda o, **kw: sr.es_ensemble(o, _FCT_MV, **kw),
        np.array([[0, 1], [1, 2], [2, 0]]),
    ),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_integer_obs_give_float_scores_on_numba(name, requires_numba):
    fn, obs = CASES[name]
    res = np.asarray(fn(obs, backend="numba"))
    assert res.dtype.kind == "f", res.dtype
    np.testing.assert_array_equal(res, fn(obs.astype(float), backend="numba"))
