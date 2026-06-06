import numpy as np
import pytest

from scoringrules._dispatch import resolve_backend_arg, use_numba
from scoringrules.backend import backends


def test_default_active_is_numpy():
    # default active must be numpy so that an active value of "numba" can only
    # come from an explicit set_active call.
    assert backends._active == "numpy"


def test_use_numba_explicit():
    assert use_numba("numba", np.asarray([1.0])) is True


def test_use_numba_none_default_is_false():
    assert use_numba(None, np.asarray([1.0])) is False


def test_use_numba_numba_with_non_numpy_raises():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="numpy-compatible"):
        use_numba("numba", torch.tensor([1.0]))


def test_legacy_backend_string_warns():
    with pytest.warns(DeprecationWarning):
        resolve_backend_arg("jax")


def test_numba_arg_does_not_warn(recwarn):
    resolve_backend_arg("numba")
    resolve_backend_arg(None)
    assert not any(isinstance(w.message, DeprecationWarning) for w in recwarn)
