import warnings

import numpy as np
import pytest

import scoringrules as sr
from scoringrules import _dispatch
from scoringrules._dispatch import resolve_backend_arg, use_numba
from scoringrules.backend import ArrayAPINamespace, backends, get_namespace
from scoringrules.backend.registry import BackendNotAvailable
from scoringrules.core.utils import GufuncWrapper

rng = np.random.default_rng(0)
OBS, FCT = rng.normal(size=4), rng.normal(size=(4, 6))
MOBS, MFCT = rng.normal(size=(4, 3)), rng.normal(size=(4, 8, 3))


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


@pytest.fixture()
def numba_missing(monkeypatch):
    """Pretend numba is not installed."""
    err = ModuleNotFoundError("No module named 'numba'", name="numba")
    monkeypatch.setattr(_dispatch, "_numba_import_error", lambda: err)
    # as after the first numba request tried the import
    monkeypatch.setattr(_dispatch, "_numba_import_cache", err)


@pytest.fixture()
def numba_broken(monkeypatch):
    """Pretend numba is installed but fails to import (e.g. numpy mismatch)."""
    err = ImportError("Numba needs NumPy 2.0 or less")
    monkeypatch.setattr(_dispatch, "_numba_import_error", lambda: err)
    monkeypatch.setattr(_dispatch, "_numba_import_cache", err)
    return err


def test_use_numba_explicit(requires_numba):
    assert use_numba("crps_ensemble", "numba", np.asarray([1.0])) is True
    # backend="numba" wins over the configuration
    with sr.config_context(numba=False):
        assert use_numba("crps_ensemble", "numba", np.asarray([1.0])) is True


def test_use_numba_numba_with_non_numpy_raises():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="numpy-compatible"):
        use_numba("crps_ensemble", "numba", torch.tensor([1.0]))


@pytest.mark.parametrize("mode", ["auto", True, False])
def test_use_numba_config_with_non_numpy_returns_false(mode):
    # the numba option only applies to numpy inputs: jax/torch take their own
    # array path silently, also with numba=True
    jnp = pytest.importorskip("jax.numpy")
    with sr.config_context(numba=mode):
        assert use_numba("crps_ensemble", None, jnp.asarray([1.0])) is False


@pytest.mark.parametrize("value", ["numpy", "jax", "torch", "foo"])
def test_use_numba_ignores_other_backend_strings(value, requires_numba):
    for mode, expected in (("auto", True), (True, True), (False, False)):
        with sr.config_context(numba=mode):
            assert use_numba("crps_ensemble", value, np.zeros(3)) is expected


def test_registry_active_name_is_not_consulted(monkeypatch, requires_numba):
    monkeypatch.setattr(backends, "_active", "numba")
    with sr.config_context(numba=False):
        assert use_numba("crps_ensemble", None, np.asarray([1.0])) is False


def test_set_active_numba_maps_to_set_config(requires_numba):
    with pytest.warns(DeprecationWarning, match=r"set_config\(numba=True\)") as rec:
        backends.set_active("numba")
    assert rec[0].filename == __file__
    assert sr.get_config()["numba"] is True
    assert backends._active == "numba"
    with pytest.warns(DeprecationWarning, match="set_active") as rec:
        backends.set_active("numpy")
    assert rec[0].filename == __file__
    assert sr.get_config()["numba"] == "auto"


@pytest.mark.parametrize("name", ["numpy", "jax", "torch"])
def test_set_active_array_api_backend_warns_and_maps_to_auto(name):
    sr.set_config(numba=False)
    with pytest.warns(DeprecationWarning, match="numba='auto'"):
        backends.set_active(name)
    assert sr.get_config()["numba"] == "auto"


def test_numba_missing_explicit_raises_import_error(numba_missing):
    with (
        pytest.warns(DeprecationWarning),
        pytest.raises(ImportError, match=r"scoringrules\[numba\]"),
    ):
        sr.crps_ensemble(np.zeros(3), np.zeros((3, 5)), backend="numba")


def test_numba_missing_numba_true_raises_import_error(numba_missing):
    with sr.config_context(numba=True):
        with pytest.raises(ImportError, match=r"numba=True requires numba"):
            sr.crps_ensemble(np.zeros(3), np.zeros((3, 5)))
        # scores without a gufunc and non-numpy inputs do not need numba
        assert sr.crps_normal(0.0, 0.0, 1.0) > 0
        torch = pytest.importorskip("torch")
        sr.crps_ensemble(torch.zeros(3), torch.zeros((3, 5)))


def test_numba_missing_set_active_raises(numba_missing):
    with (
        pytest.warns(DeprecationWarning),
        pytest.raises(BackendNotAvailable, match=r"scoringrules\[numba\]"),
    ):
        backends.set_active("numba")
    assert sr.get_config()["numba"] == "auto"
    # both missing-numba paths are caught by ``except ImportError``
    assert issubclass(BackendNotAvailable, ImportError)


def test_numba_missing_auto_is_silent(numba_missing, recwarn, gufunc_calls):
    with sr.config_context(numba="auto"):
        assert use_numba("crps_ensemble", None, np.zeros(3), np.zeros((3, 5))) is False
        out = sr.crps_ensemble(np.zeros(3), np.zeros((3, 5)))
        sr.es_ensemble(np.zeros((3, 2)), np.zeros((3, 5, 2)))
    np.testing.assert_allclose(out, 0.0)
    assert gufunc_calls == []
    assert len(recwarn) == 0


def test_numba_broken_reports_and_chains_import_error(numba_broken):
    with (
        pytest.warns(DeprecationWarning),
        pytest.raises(ImportError, match="could not be imported") as info,
    ):
        sr.crps_ensemble(np.zeros(3), np.zeros((3, 5)), backend="numba")
    assert "NumPy 2.0" in str(info.value)
    assert "pip install" not in str(info.value)
    assert info.value.__cause__ is numba_broken
    with (
        pytest.warns(DeprecationWarning),
        pytest.raises(BackendNotAvailable, match="could not be imported") as info,
    ):
        backends.set_active("numba")
    assert info.value.__cause__ is numba_broken
    assert sr.get_config()["numba"] == "auto"


def test_numba_broken_auto_warns_once(numba_broken, monkeypatch):
    monkeypatch.setattr(_dispatch, "_broken_numba_warned", False)
    obs, fct = np.zeros(3), np.zeros((3, 5))
    with sr.config_context(numba="auto"):
        with pytest.warns(RuntimeWarning, match="NumPy 2.0") as rec:
            out = sr.crps_ensemble(obs, fct)
        assert rec[0].filename == __file__
        np.testing.assert_allclose(out, 0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sr.crps_ensemble(obs, fct)


def test_numba_broken_not_listed(numba_broken):
    # once the import failed, the registry uses its result like ``backend="numba"``
    assert "numba" not in backends.available_backends
    assert "numba" not in type(backends)()
    with pytest.raises(BackendNotAvailable, match="could not be imported"):
        backends.register_backend("numba")


def test_numba_available_before_import_only_checks_installed(monkeypatch):
    # ``import scoringrules`` lists numba without importing it
    monkeypatch.setattr(_dispatch, "_numba_import_cache", _dispatch._UNSET)
    monkeypatch.setattr(_dispatch, "find_spec", lambda name: None)
    assert _dispatch._numba_available() is False
    monkeypatch.setattr(_dispatch, "find_spec", lambda name: object())
    assert _dispatch._numba_available() is True


def test_failed_numba_import_unlists_numba(monkeypatch):
    # the registry lists numba at import without importing it; a failed import
    # on the first numba request takes it out again
    import sys

    monkeypatch.setattr(_dispatch, "_numba_import_cache", _dispatch._UNSET)
    monkeypatch.setitem(sys.modules, "numba", None)  # ``import numba`` fails
    had = dict.get(backends, "numba")
    dict.__setitem__(backends, "numba", backends._get("numpy"))
    try:
        assert isinstance(_dispatch._numba_import_error(), ImportError)
        assert "numba" not in backends
        assert "numba" not in backends.available_backends
    finally:
        if had is not None:
            dict.__setitem__(backends, "numba", had)


# --- the deprecated registry shim ---------------------------------------------


def test_registry_item_returns_namespace_and_warns(backend):
    with pytest.warns(DeprecationWarning, match="removed in 1.0"):
        B = sr.backends[backend]
    assert isinstance(B, ArrayAPINamespace)
    x = {"numpy": np, "numba": np}.get(backend)
    if x is None:
        x = pytest.importorskip("jax.numpy" if backend == "jax" else backend)
    arr = x.asarray([1.0, 2.0, 3.0])
    assert B._xp is get_namespace(arr)._xp
    # old ``B = sr.backends[...]; B.op(...)`` code keeps working for standard ops
    assert float(B.sum(arr, axis=-1)) == 6.0
    assert float(B.erf(B.asarray(0.0))) == 0.0


def test_registry_numba_maps_to_numpy(requires_numba):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert sr.backends["numba"] is sr.backends["numpy"]


def test_registry_active_warns():
    with pytest.warns(DeprecationWarning, match="active"):
        B = backends.active
    assert B._xp is get_namespace(np.zeros(1))._xp


def test_registry_active_follows_set_active(requires_numba):
    numpy_xp = get_namespace(np.zeros(1))._xp
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        backends.set_active("numba")
        assert backends.active is backends["numba"]
        assert backends.active._xp is numpy_xp
        jnp = pytest.importorskip("jax.numpy")
        backends.set_active("jax")
        assert backends.active._xp is get_namespace(jnp.zeros(1))._xp
        backends.set_active("numpy")
        assert backends.active._xp is numpy_xp
    assert sr.get_config()["numba"] == "auto"


@pytest.mark.parametrize(
    "access",
    [
        lambda r: r.get("numpy"),
        lambda r: list(r.values()),
        lambda r: list(r.items()),
    ],
    ids=["get", "values", "items"],
)
def test_registry_value_accessors_warn(access):
    with pytest.warns(DeprecationWarning, match="removed in 1.0"):
        out = access(backends)
    assert out


def test_registry_key_lookups_are_silent(recwarn):
    assert "numpy" in backends
    assert "numpy" in list(backends)
    assert "numpy" in backends.keys()
    assert "numpy" in backends.available_backends
    assert not any(isinstance(w.message, DeprecationWarning) for w in recwarn)


def test_registry_unavailable_backend_raises(monkeypatch):
    monkeypatch.setattr(
        type(backends), "available_backends", property(lambda self: ["numpy"])
    )
    with pytest.raises(BackendNotAvailable, match="torch"):
        backends.register_backend("torch")


@pytest.mark.parametrize("name", ["numpy", "jax", "torch"])
def test_register_backend_warns(name):
    if name != "numpy":
        pytest.importorskip(name)
    with pytest.warns(DeprecationWarning, match="register_backend"):
        sr.register_backend(name)


def test_register_backend_numba_does_not_warn(requires_numba, recwarn):
    sr.register_backend("numba")
    assert not any(isinstance(w.message, DeprecationWarning) for w in recwarn)


# --- the deprecated backend argument ---------------------------------------------


@pytest.mark.parametrize("value", ["numpy", "jax", "torch", "numba", "foo"])
def test_backend_arg_warns(value):
    with pytest.warns(DeprecationWarning, match="removed in 1.0"):
        out = resolve_backend_arg(value)
    assert out is (_dispatch.FORWARDED_NUMBA if value == "numba" else None)


def test_backend_arg_none_and_forwarded_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve_backend_arg(None) is None
        forwarded = _dispatch.FORWARDED_NUMBA
        assert resolve_backend_arg(forwarded) is forwarded


_DEPRECATION_CALLS = {
    "crps_ensemble": lambda b: sr.crps_ensemble(OBS, FCT, backend=b),
    "twcrps_ensemble": lambda b: sr.twcrps_ensemble(OBS, FCT, backend=b),
    "twgksuv_ensemble": lambda b: sr.twgksuv_ensemble(OBS, FCT, backend=b),
    "twes_ensemble": lambda b: sr.twes_ensemble(MOBS, MFCT, lambda x: x, backend=b),
    "twvs_ensemble": lambda b: sr.twvs_ensemble(MOBS, MFCT, lambda x: x, backend=b),
    "twgksmv_ensemble": lambda b: sr.twgksmv_ensemble(
        MOBS, MFCT, lambda x: x, backend=b
    ),
    "crps_normal": lambda b: sr.crps_normal(0.0, 0.0, 1.0, backend=b),
    "energy_score": lambda b: sr.energy_score(MOBS, MFCT, backend=b),
}


@pytest.mark.parametrize("value", ["numba", "numpy"])
@pytest.mark.parametrize("name", list(_DEPRECATION_CALLS))
def test_backend_arg_warns_once_at_the_user_call(name, value, request):
    # the threshold-weighted scores forward backend to the score they chain to
    # without a second warning; the warning points at this file, also through
    # the deprecated aliases
    if value == "numba":
        request.getfixturevalue("requires_numba")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _DEPRECATION_CALLS[name](value)
    backend_warnings = [w for w in rec if "backend=" in str(w.message)]
    assert len(backend_warnings) == 1, [str(w.message) for w in rec]
    assert issubclass(backend_warnings[0].category, DeprecationWarning)
    assert f"backend={value!r}" in str(backend_warnings[0].message)
    assert backend_warnings[0].filename == __file__


@pytest.mark.parametrize(
    "name", ["twcrps_ensemble", "twgksuv_ensemble", "twes_ensemble", "twvs_ensemble"]
)
def test_backend_numba_forwarded_runs_the_gufunc(name, gufunc_calls, requires_numba):
    with pytest.warns(DeprecationWarning), sr.config_context(numba=False):
        _DEPRECATION_CALLS[name]("numba")
    assert len(gufunc_calls) == 1


def test_backend_numba_is_a_per_call_numba_true(gufunc_calls, requires_numba):
    with sr.config_context(numba=False), pytest.warns(DeprecationWarning):
        res = sr.crps_ensemble(OBS, FCT, backend="numba")
    assert len(gufunc_calls) == 1
    with sr.config_context(numba=True):
        np.testing.assert_array_equal(res, sr.crps_ensemble(OBS, FCT))
