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


# --- lazy loading, exact-dtype loops ---------------------------------------------

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import textwrap  # noqa: E402


def _run_fresh(code):
    """Run ``code`` in a fresh interpreter (inheriting NUMBA_CACHE_DIR) and return
    the JSON it prints on its last line."""
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_import_does_not_import_numba():
    res = _run_fresh(
        """
        import json, sys
        from importlib.util import find_spec
        import scoringrules as sr
        listed = "numba" in sr.backends.available_backends
        print(json.dumps({
            "numba": "numba" in sys.modules,
            "gufuncs": sorted(m for m in sys.modules if "_gufunc" in m),
            "listed": listed,
            "installed": find_spec("numba") is not None,
        }))
        """
    )
    assert res["numba"] is False
    assert res["gufuncs"] == []
    # listing numba as available must not import it either
    assert res["listed"] == res["installed"]


# One call per kind of gufunc argument: univariate, weighted, multivariate, a
# ``()`` Python scalar (dss ``bias``, vs ``p``), the vs weight matrix, 0-d obs.
_DTYPE_CASES = """
import numpy as np
import scoringrules as sr

def cases(dt):
    rng = np.random.default_rng(1)
    o, f = rng.normal(size=4).astype(dt), rng.normal(size=(4, 7)).astype(dt)
    om, fm = rng.normal(size=(4, 3)).astype(dt), rng.normal(size=(4, 7, 3)).astype(dt)
    w = rng.uniform(0.5, 1.5, size=(4, 7)).astype(dt)
    nb = {"backend": "numba"}
    return {
        "crps_pwm": lambda: sr.crps_ensemble(o, f, estimator="pwm", **nb),
        "crps_nrg": lambda: sr.crps_ensemble(o, f, estimator="nrg", **nb),
        "crps_w_qd": lambda: sr.crps_ensemble(o, f, ens_w=w, estimator="qd", **nb),
        "owcrps_0d": lambda: sr.owcrps_ensemble(o[0], f[0], a=-0.5, **nb),
        "es_nrg": lambda: sr.es_ensemble(om, fm, **nb),
        "dssuv": lambda: sr.dssuv_ensemble(o, f, bias=True, **nb),
        "vs_p": lambda: sr.vs_ensemble(om, fm, p=1.5, **nb),
        "vs_w": lambda: sr.vs_ensemble(om, fm, w=np.ones((3, 3), dtype=dt), **nb),
    }
"""


def test_float32_after_float64_runs_a_float32_loop(requires_numba):
    ns = {}
    exec(_DTYPE_CASES, ns)  # noqa: S102
    fresh = _run_fresh(
        _DTYPE_CASES
        + """
import json
print(json.dumps({k: np.asarray(fn()).tolist() for k, fn in cases(np.float32).items()}))
"""
    )
    for name, fn in ns["cases"](np.float64).items():
        assert np.asarray(fn()).dtype == np.float64, name
    for name, fn in ns["cases"](np.float32).items():
        res = np.asarray(fn())
        assert res.dtype == np.float32, name
        # bit-identical to a process that never compiled a float64 loop
        np.testing.assert_array_equal(res, np.asarray(fresh[name], np.float32), name)


def test_float32_loops_exist_after_float64(requires_numba):
    from scoringrules.core.crps import _gufuncs, estimator_gufuncs

    rng = np.random.default_rng(3)
    obs, fct = rng.normal(size=3), rng.normal(size=(3, 5))
    for dt in (np.float64, np.float32):
        sr.crps_ensemble(
            obs.astype(dt), fct.astype(dt), estimator="akr", backend="numba"
        )
    assert estimator_gufuncs("akr").__wrapped__ is _gufuncs._crps_ensemble_akr_gufunc
    types = _gufuncs._crps_ensemble_akr_gufunc.ufunc.types
    assert "ff->f" in types and "dd->d" in types


def test_dssmv_float32_in_fresh_process(requires_numba):
    res = _run_fresh(
        """
        import json
        import numpy as np
        import scoringrules as sr
        rng = np.random.default_rng(2)
        obs, fct = rng.normal(size=(4, 3)), rng.normal(size=(4, 9, 3))
        r32 = sr.dssmv_ensemble(obs.astype(np.float32), fct.astype(np.float32),
                                backend="numba")
        r64 = sr.dssmv_ensemble(obs, fct)
        print(json.dumps({"dtype": str(r32.dtype), "r32": r32.tolist(),
                          "r64": r64.tolist()}))
        """
    )
    assert res["dtype"] == "float32"
    np.testing.assert_allclose(res["r32"], res["r64"], rtol=1e-4)


_RNG_B = np.random.default_rng(7)
_OBS_B, _FCT_B = _RNG_B.normal(size=4) * 3, _RNG_B.normal(size=(4, 6)) * 3


# Neither input floating gives float64, as on the array path (main gave float32
# for bool and small integers). Mixed float/integer inputs keep numpy's promotion.
@pytest.mark.parametrize(
    "fn",
    [sr.dssuv_ensemble, sr.error_spread_score],
    ids=["dssuv_ensemble", "error_spread_score"],
)
@pytest.mark.parametrize(
    "dtypes",
    [("bool", "bool"), ("bool", "float64"), ("int8", "int8"), ("uint8", "int16")],
    ids=["bool", "bool_obs", "int8", "uint8_int16"],
)
def test_non_float_output_dtype_matches_array_path(fn, dtypes, requires_numba):
    obs = (_OBS_B > 0) if dtypes[0] == "bool" else np.abs(_OBS_B)
    # two True members per row, so the ensemble spread is not zero
    fct = (
        (np.arange(24).reshape(4, 6) % 3 == 0)
        if dtypes[1] == "bool"
        else np.abs(_FCT_B)
    )
    obs, fct = obs.astype(dtypes[0]), fct.astype(dtypes[1])
    res_numba = np.asarray(fn(obs, fct, backend="numba"))
    res_array = np.asarray(fn(obs, fct))
    assert res_numba.dtype == res_array.dtype == np.float64
    np.testing.assert_allclose(res_numba, res_array, rtol=1e-12, equal_nan=True)
