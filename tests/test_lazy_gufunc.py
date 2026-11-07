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


def test_float32_stays_float32_on_numba(requires_numba):
    f32 = np.float32
    fct = _FCT_UV.astype(f32)
    # 0-d obs: a numpy float32 scalar, not a Python float
    res = sr.owcrps_ensemble(f32(0.3), fct[0], -1.0, 1.0, backend="numba")
    assert isinstance(res, np.float32)
    # auxiliary 0-d arrays built by the caller (w_median) do not promote
    args = [
        np.asarray(x, dtype=f32)
        for x in ([0.1, 0.5], [0.0, 0.2], [[-1, -0.5]] * 2, [[1, 0.5]] * 2, [0.2, 0.5])
    ]
    assert sr.weighted_interval_score(*args, backend="numba").dtype == f32
