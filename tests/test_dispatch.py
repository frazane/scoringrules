import warnings

import numpy as np
import pytest

import scoringrules as sr
from scoringrules import _dispatch
from scoringrules._dispatch import resolve_backend_arg, use_numba
from scoringrules.backend import ArrayAPINamespace, backends, get_namespace
from scoringrules.backend.registry import BackendNotAvailable


@pytest.fixture()
def numba_missing(monkeypatch):
    """Pretend numba is not installed."""
    monkeypatch.setattr(_dispatch, "_numba_installed", lambda: False)


def test_default_numba_flag_is_off():
    # the global numba switch can only be turned on by an explicit set_active call
    assert _dispatch._numba_active is False
    assert backends._active == "numpy"


def test_use_numba_explicit(requires_numba):
    assert use_numba("numba", np.asarray([1.0])) is True


def test_use_numba_none_default_is_false():
    assert use_numba(None, np.asarray([1.0])) is False


def test_use_numba_numba_with_non_numpy_raises():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="numpy-compatible"):
        use_numba("numba", torch.tensor([1.0]))


def test_use_numba_reads_only_the_dispatch_flag(monkeypatch, requires_numba):
    monkeypatch.setattr(_dispatch, "_numba_active", True)
    assert use_numba(None, np.asarray([1.0])) is True
    # the registry's name is not consulted any more
    monkeypatch.setattr(_dispatch, "_numba_active", False)
    monkeypatch.setattr(backends, "_active", "numba")
    assert use_numba(None, np.asarray([1.0])) is False


def test_use_numba_global_numba_with_non_numpy_returns_false(monkeypatch):
    # set_active("numba") globally must NOT force the gufunc path on jax/torch
    # inputs (it only applies to numpy inputs); it returns False, not a raise.
    jnp = pytest.importorskip("jax.numpy")
    monkeypatch.setattr(_dispatch, "_numba_active", True)
    assert use_numba(None, jnp.asarray([1.0])) is False


def test_set_active_numba_toggles_flag(restore_active, requires_numba, recwarn):
    backends.set_active("numba")
    assert _dispatch._numba_active is True
    assert not any(isinstance(w.message, DeprecationWarning) for w in recwarn)
    with pytest.warns(DeprecationWarning, match="set_active"):
        backends.set_active("numpy")
    assert _dispatch._numba_active is False


@pytest.mark.parametrize("name", ["numpy", "jax", "torch"])
def test_set_active_array_api_backend_warns_and_clears_flag(
    name, restore_active, monkeypatch
):
    monkeypatch.setattr(_dispatch, "_numba_active", True)
    with pytest.warns(DeprecationWarning, match="set_active"):
        backends.set_active(name)
    assert _dispatch._numba_active is False


def test_numba_missing_explicit_raises_import_error(numba_missing):
    with pytest.raises(ImportError, match=r"scoringrules\[numba\]"):
        sr.crps_ensemble(np.zeros(3), np.zeros((3, 5)), backend="numba")


def test_numba_missing_global_flag_raises_import_error(numba_missing, monkeypatch):
    monkeypatch.setattr(_dispatch, "_numba_active", True)
    with pytest.raises(ImportError, match="set_active"):
        use_numba(None, np.asarray([1.0]))


def test_numba_missing_set_active_raises(numba_missing, restore_active):
    with pytest.raises(BackendNotAvailable, match=r"scoringrules\[numba\]"):
        backends.set_active("numba")
    assert _dispatch._numba_active is False


def test_numba_missing_default_path_unaffected(numba_missing):
    out = sr.crps_ensemble(np.zeros(3), np.zeros((3, 5)))
    np.testing.assert_allclose(out, 0.0)


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


def test_registry_active_warns(restore_active):
    with pytest.warns(DeprecationWarning, match="active"):
        B = backends.active
    assert B._xp is get_namespace(np.zeros(1))._xp


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


def test_legacy_backend_string_warns():
    with pytest.warns(DeprecationWarning):
        resolve_backend_arg("jax")


def test_numba_arg_does_not_warn(recwarn):
    resolve_backend_arg("numba")
    resolve_backend_arg(None)
    assert not any(isinstance(w.message, DeprecationWarning) for w in recwarn)
