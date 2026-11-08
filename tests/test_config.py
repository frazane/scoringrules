"""The configuration API (``set_config``, ``get_config``, ``config_context``) and
the numba modes it selects."""

import asyncio
import json
import os
import subprocess
import sys
import textwrap
import threading
import warnings

import numpy as np
import pytest

import scoringrules as sr
from scoringrules import _config, _dispatch
from scoringrules._dispatch import use_numba
from scoringrules.core.utils import GufuncWrapper

rng = np.random.default_rng(0)
OBS, FCT = rng.normal(size=4), rng.normal(size=(4, 6))


@pytest.fixture()
def gufunc_calls(monkeypatch):
    """The gufunc calls (every gufunc goes through ``GufuncWrapper``)."""
    calls = []
    call = GufuncWrapper.__call__

    def counting(self, *args):
        calls.append(self.__name__)
        return call(self, *args)

    monkeypatch.setattr(GufuncWrapper, "__call__", counting)
    return calls


def _numba():
    return sr.get_config()["numba"]


# --- get_config / set_config / config_context ------------------------------------


def test_exported():
    for name in ("set_config", "get_config", "config_context"):
        assert name in sr.__all__


def test_default_is_auto(monkeypatch):
    monkeypatch.delenv("SCORINGRULES_NUMBA", raising=False)
    assert _config._numba_from_env() == "auto"


def test_get_config_returns_a_copy():
    config = sr.get_config()
    config["numba"] = False
    assert sr.get_config() != config


@pytest.mark.parametrize("value", ["auto", True, False])
def test_set_config(value):
    sr.set_config(numba=value)
    assert _numba() == value and type(_numba()) is type(value)
    sr.set_config()  # None keeps the value
    assert _numba() == value


@pytest.mark.parametrize(
    "value", ["AUTO", "true", 1, 0, 1.0, np.True_, np.array(True), "numba", []]
)
def test_invalid_values_raise(value):
    before = sr.get_config()
    with pytest.raises(ValueError, match="numba must be"):
        sr.set_config(numba=value)
    with pytest.raises(ValueError, match="numba must be"):
        with sr.config_context(numba=value):
            pass
    assert sr.get_config() == before


def test_keyword_only():
    with pytest.raises(TypeError):
        sr.set_config(True)
    with pytest.raises(TypeError):
        sr.config_context(True)


def test_config_context_nests_and_restores():
    sr.set_config(numba="auto")
    with sr.config_context(numba=False):
        assert _numba() is False
        with sr.config_context(numba=True):
            assert _numba() is True
            with sr.config_context():  # no change
                assert _numba() is True
        assert _numba() is False
    assert _numba() == "auto"


def test_config_context_restores_on_exception():
    with pytest.raises(KeyError):
        with sr.config_context(numba=False):
            raise KeyError
    assert _numba() == "auto"


def test_config_context_as_decorator():
    @sr.config_context(numba=False)
    def inner():
        return _numba()

    assert inner() is False
    assert inner() is False  # reusable
    assert _numba() == "auto"


def test_set_config_inside_a_context_is_local():
    # set_config changes the innermost context, and is undone with it
    with sr.config_context(numba=False):
        sr.set_config(numba=True)
        assert _numba() is True
        with sr.config_context():
            sr.set_config(numba="auto")
            assert _numba() == "auto"
        assert _numba() is True
    assert _numba() == "auto"
    assert _config._global_config["numba"] == "auto"


def test_set_config_outside_a_context_is_global():
    sr.set_config(numba=False)
    assert _config._global_config["numba"] is False
    with sr.config_context(numba=True):
        assert _numba() is True
    assert _numba() is False


def _in_thread(fn):
    out = []
    t = threading.Thread(target=lambda: out.append(fn()))
    t.start()
    t.join()
    return out[0]


# free-threaded builds (and -X thread_inherit_context=1) start threads with a
# copy of the creating thread's context
_THREADS_INHERIT = bool(getattr(sys.flags, "thread_inherit_context", False))


def test_threads_see_the_global_value_not_other_contexts():
    sr.set_config(numba=False)
    with sr.config_context(numba=True):
        # a new thread starts without this thread's context (or with a copy)
        assert _in_thread(_numba) is _THREADS_INHERIT
    # a thread's own context does not leak into this one
    started, release = threading.Event(), threading.Event()

    def worker():
        with sr.config_context(numba=True):
            started.set()
            release.wait(5)
            return _numba()

    out = []
    t = threading.Thread(target=lambda: out.append(worker()))
    t.start()
    started.wait(5)
    assert _numba() is False
    release.set()
    t.join()
    assert out == [True]
    # set_config outside any context is seen by every thread
    sr.set_config(numba="auto")
    assert _in_thread(_numba) == "auto"


def test_asyncio_tasks_are_isolated():
    async def task(value, seen):
        with sr.config_context(numba=value):
            for _ in range(3):
                await asyncio.sleep(0)  # interleave with the other tasks
                seen.append((value, _numba()))

    async def inherits():
        return _numba()

    async def main():
        seen = []
        await asyncio.gather(*(task(v, seen) for v in ("auto", True, False)))
        with sr.config_context(numba=False):
            # a task copies the context it was created in
            child = asyncio.create_task(inherits())
        return seen, await child, _numba()

    seen, child, after = asyncio.run(main())
    assert len(seen) == 9
    assert all(value is seen_value for value, seen_value in seen)
    assert child is False
    assert after == "auto"


# --- the environment variable ------------------------------------------------------


@pytest.mark.parametrize(
    "raw,value",
    [
        ("0", False),
        ("false", False),
        ("False", False),
        (" 1 ", True),
        ("true", True),
        ("TRUE", True),
        ("auto", "auto"),
        ("Auto", "auto"),
        ("", "auto"),
    ],
)
def test_env_var_parsing(raw, value, monkeypatch):
    monkeypatch.setenv("SCORINGRULES_NUMBA", raw)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = _config._numba_from_env()
    assert result == value and type(result) is type(value)


@pytest.mark.parametrize("raw", ["yes", "2", "numba", "none"])
def test_env_var_invalid_warns_and_gives_auto(raw, monkeypatch):
    monkeypatch.setenv("SCORINGRULES_NUMBA", raw)
    with pytest.warns(UserWarning, match="SCORINGRULES_NUMBA"):
        assert _config._numba_from_env() == "auto"


def _run_fresh(code, **env):
    """Run ``code`` in a fresh interpreter with ``env`` added to the environment
    and return the JSON it prints on its last line, and its stderr."""
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, **env},
    )
    return json.loads(out.stdout.strip().splitlines()[-1]), out.stderr


_PRINT_CONFIG = """
import json, sys
import scoringrules as sr
print(json.dumps([sr.get_config()["numba"], "numba" in sys.modules]))
"""


@pytest.mark.parametrize("raw,value", [("0", False), ("1", True), ("auto", "auto")])
def test_env_var_read_at_import(raw, value):
    (numba, imported), _ = _run_fresh(_PRINT_CONFIG, SCORINGRULES_NUMBA=raw)
    assert numba == value
    assert imported is False  # numba=True does not import numba at import


def test_env_var_invalid_at_import_warns():
    (numba, _), stderr = _run_fresh(_PRINT_CONFIG, SCORINGRULES_NUMBA="bogus")
    assert numba == "auto"
    assert "UserWarning" in stderr and "SCORINGRULES_NUMBA" in stderr


def test_auto_imports_numba_only_for_a_gufunc_call(requires_numba):
    (res, _) = _run_fresh(
        """
        import json, sys
        import numpy as np
        import scoringrules as sr
        seen = {}
        obs, fct = np.zeros(3), np.ones((3, 5))
        sr.crps_normal(0.0, 0.0, 1.0)  # no gufunc
        seen["no_gufunc"] = "numba" in sys.modules
        sr.crps_ensemble(np.zeros(3, np.float16), fct)  # guard: array path
        seen["float16"] = "numba" in sys.modules
        with sr.config_context(numba=False):
            sr.crps_ensemble(obs, fct)
        seen["numba_false"] = "numba" in sys.modules
        sr.weighted_interval_score(0.1, 0.0, [-1.0], [1.0], [0.2])  # excluded
        seen["excluded"] = "numba" in sys.modules
        sr.crps_ensemble(obs, fct)
        seen["gufunc"] = "numba" in sys.modules
        print(json.dumps(seen))
        """,
        SCORINGRULES_NUMBA="auto",
    )
    assert res == {
        "no_gufunc": False,
        "float16": False,
        "numba_false": False,
        "excluded": False,
        "gufunc": True,
    }


# --- the numba modes --------------------------------------------------------------


@pytest.mark.parametrize("mode,n_calls", [("auto", 1), (True, 1), (False, 0)])
def test_modes_on_numpy_input(mode, n_calls, gufunc_calls, requires_numba):
    with sr.config_context(numba=False):
        ref = sr.crps_ensemble(OBS, FCT)
    with sr.config_context(numba=mode):
        res = sr.crps_ensemble(OBS, FCT)
    assert len(gufunc_calls) == n_calls
    np.testing.assert_allclose(res, ref, rtol=1e-12)


@pytest.mark.parametrize("mode", ["auto", True, False])
def test_modes_on_torch_input_take_the_array_path(mode, gufunc_calls):
    torch = pytest.importorskip("torch")
    with sr.config_context(numba=mode), warnings.catch_warnings():
        warnings.simplefilter("error")
        res = sr.crps_ensemble(torch.as_tensor(OBS), torch.as_tensor(FCT))
    assert isinstance(res, torch.Tensor)
    assert gufunc_calls == []


def _wis_args(dtype=np.float64):
    return [
        np.asarray(x, dtype=dtype)
        for x in ([0.1, 0.5], [0.0, 0.2], [[-1, -0.5]] * 2, [[1, 0.5]] * 2, [0.2, 0.5])
    ]


_EXCLUSION_CASES = {
    "wis": (sr.weighted_interval_score, _wis_args(), {}, True),
    "wis_f32": (sr.weighted_interval_score, _wis_args(np.float32), {}, True),
    **{
        f"gksuv_{e}_{dt}": (
            sr.gksuv_ensemble,
            (OBS.astype(dt), FCT.astype(dt)),
            {"estimator": e},
            dt == "float32" and e.startswith("akr"),
        )
        for e in ("nrg", "fair", "akr", "akr_circperm")
        for dt in ("float32", "float64")
    },
    # the float dtype follows numpy's promotion of obs and fct
    "gksuv_akr_f32_python_obs": (
        sr.gksuv_ensemble,
        (0.5, FCT[0].astype(np.float32)),
        {"estimator": "akr"},
        True,
    ),
    "gksuv_akr_f32_big_endian": (
        sr.gksuv_ensemble,
        (OBS.astype(">f4"), FCT.astype(">f4")),
        {"estimator": "akr"},
        True,
    ),
    "gksuv_akr_mixed": (
        sr.gksuv_ensemble,
        (OBS, FCT.astype(np.float32)),
        {"estimator": "akr"},
        False,
    ),
    "gksuv_akr_int": (
        sr.gksuv_ensemble,
        (np.arange(4), np.ones((4, 6), int)),
        {"estimator": "akr"},
        False,
    ),
    "twgksuv_akr_f32": (
        sr.twgksuv_ensemble,
        (OBS.astype(np.float32), FCT.astype(np.float32)),
        {"estimator": "akr"},
        True,
    ),
    "crps_akr_f32": (
        sr.crps_ensemble,
        (OBS.astype(np.float32), FCT.astype(np.float32)),
        {"estimator": "akr"},
        False,
    ),
}


@pytest.mark.parametrize("case", list(_EXCLUSION_CASES))
def test_auto_exclusions(case, gufunc_calls, requires_numba):
    fn, args, kwargs, excluded = _EXCLUSION_CASES[case]
    with sr.config_context(numba="auto"):
        fn(*args, **kwargs)
    assert len(gufunc_calls) == (0 if excluded else 1)
    # numba=True still reaches them (and the parity tests cover them)
    with sr.config_context(numba=True):
        fn(*args, **kwargs)
    assert len(gufunc_calls) == (1 if excluded else 2)


def _estimators():
    """The estimators each estimator-keyed score validates its input against."""
    from scoringrules import _crps, _energy, _kernels, _variogram

    return {
        "crps_ensemble": _crps._CRPS_ESTIMATORS,
        "gksuv_ensemble": _kernels._KS_ESTIMATORS,
        "gksmv_ensemble": _kernels._KS_ESTIMATORS,
        "es_ensemble": _energy._ES_ESTIMATORS,
        "vs_ensemble": _variogram._VS_ESTIMATORS,
    }


def test_exclusion_keys_are_scores_with_a_gufunc():
    estimators = _estimators()
    for key in _dispatch.AUTO_EXCLUDED:
        if isinstance(key, str):
            name = key
            assert name not in estimators, f"{key}: key it by (name, estimator)"
        else:
            name, estimator = key
            assert name in estimators, f"{name} does not take an estimator"
            assert estimator in estimators[name], key
        assert callable(getattr(sr, name))


def test_deprecated_alias_warns_at_the_caller():
    obs, fct = rng.normal(size=(4, 3)), rng.normal(size=(4, 6, 3))
    with pytest.warns(DeprecationWarning, match="energy_score is deprecated") as rec:
        res = sr.energy_score(obs, fct)
    assert [w.filename for w in rec] == [__file__]
    np.testing.assert_array_equal(res, sr.es_ensemble(obs, fct))


def test_use_numba_modes_unit(requires_numba):
    x = np.zeros(3)
    for mode, expected in (("auto", True), (True, True), (False, False)):
        with sr.config_context(numba=mode):
            assert use_numba("crps_ensemble", None, x) is expected
            assert use_numba(("gksuv_ensemble", "akr"), None, x) is expected
            x32 = x.astype(np.float32)
            excluded = use_numba(("gksuv_ensemble", "akr"), None, x32, x32)
            assert excluded is (mode is True)
