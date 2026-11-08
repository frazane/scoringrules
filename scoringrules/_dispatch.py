# scoringrules/_dispatch.py
"""Backend-argument handling for the array-API era.

``backend`` historically chose both the array library and numba. The library is
now inferred from the input, so the only meaningful values are ``None`` (infer)
and ``"numba"`` (gufunc fast-path). The legacy framework strings and the
registry remain accepted, with a DeprecationWarning, until 1.0.

The global numba switch lives here as a private flag. Its public setter is
``scoringrules.backends.set_active("numba")`` (and ``set_active("numpy")`` to
reset it), which refuses to enable it when numba cannot be imported."""

import warnings
from importlib.util import find_spec

from array_api_compat import is_array_api_obj

_LEGACY_ARRAY_API_BACKENDS = {"numpy", "jax", "torch"}

NUMBA_INSTALL_HINT = (
    "Install it with `pip install numba` (or `pip install scoringrules[numba]`)."
)

# Set by ``backends.set_active``; read only by ``use_numba``.
_numba_active = False
_UNSET = object()
_numba_import_cache = _UNSET


def _set_numba_active(flag):
    global _numba_active
    _numba_active = bool(flag)


def _numba_import_error():
    """``None`` if numba imports, else the ``ImportError`` it raised (cached).

    This is the real import (about 0.2 s), done on the first numba request."""
    global _numba_import_cache
    if _numba_import_cache is _UNSET:
        try:
            import numba  # noqa: F401
        except ImportError as err:
            _numba_import_cache = err
            # the registry listed numba as installed at import: unlist it
            from scoringrules.backend import backends

            dict.pop(backends, "numba", None)
        else:
            _numba_import_cache = None
    return _numba_import_cache


def _numba_available():
    """Whether numba can be used, without importing it.

    Until the first numba request imports it (``_numba_import_error``), this is
    only a check that numba is installed; afterwards it is the import result."""
    if _numba_import_cache is _UNSET:
        try:
            return find_spec("numba") is not None
        except ValueError:  # numba in sys.modules without a spec
            return True
    return _numba_import_cache is None


def numba_unavailable_error(what, cls=ImportError):
    """The exception to raise when ``what`` needs numba but numba cannot be used.

    A missing numba gets the install hint; a numba that is installed but fails
    to import (e.g. a numba/numpy version mismatch) reports and chains the
    original error instead."""
    err = _numba_import_error()
    missing = isinstance(err, ModuleNotFoundError) and err.name == "numba"
    if missing:
        exc = cls(
            f"{what} requires numba, which is not installed. {NUMBA_INSTALL_HINT}"
        )
    else:
        exc = cls(f"{what} requires numba, but it could not be imported: {err}")
        exc.__cause__ = err
    exc.__suppress_context__ = True
    return exc


def resolve_backend_arg(backend):
    """Warn on deprecated backend strings; return the value unchanged."""
    if backend in _LEGACY_ARRAY_API_BACKENDS:
        warnings.warn(
            f"Passing backend={backend!r} is deprecated and will be removed in "
            "1.0. The array framework is now inferred from the input; remove the "
            "backend argument (use backend='numba' only for the numba fast-path).",
            DeprecationWarning,
            stacklevel=3,
        )
    return backend


def _is_numpy_compatible(*arrays):
    for a in arrays:
        if is_array_api_obj(a):
            name = type(a).__module__
            if not name.startswith("numpy"):
                return False
    return True


def use_numba(backend, *arrays):
    """Decide whether to use the numba gufunc path for this call.

    True iff ``backend == "numba"`` (explicit), or ``backend is None`` and the
    user globally selected numba via ``set_active("numba")`` AND the inputs are
    numpy-compatible. A numba request with non-numpy inputs is an error, and so
    is a numba request when numba cannot be imported (``ImportError``).
    """
    explicit = backend == "numba"
    global_numba = backend is None and _numba_active
    if not (explicit or global_numba):
        return False
    if not _is_numpy_compatible(*arrays):
        if explicit:
            raise ValueError(
                "backend='numba' requires numpy-compatible inputs; pass numpy "
                "arrays or drop backend='numba' to use the input's framework."
            )
        return False  # global numba default does not apply to jax/torch inputs
    if _numba_import_error() is not None:
        how = "backend='numba'" if explicit else "set_active('numba')"
        raise numba_unavailable_error(how)
    return True
