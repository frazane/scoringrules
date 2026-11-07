# scoringrules/_dispatch.py
"""Backend-argument handling for the array-API era.

``backend`` historically chose both the array library and numba. The library is
now inferred from the input, so the only meaningful values are ``None`` (infer)
and ``"numba"`` (gufunc fast-path). The legacy framework strings and the
registry remain accepted, with a DeprecationWarning, until 1.0.

The global numba switch lives here as a private flag. Its public setter is
``scoringrules.backends.set_active("numba")`` (and ``set_active("numpy")`` to
reset it), which refuses to enable it when numba is not installed."""

import warnings

from array_api_compat import is_array_api_obj

_LEGACY_ARRAY_API_BACKENDS = {"numpy", "jax", "torch"}

NUMBA_INSTALL_HINT = (
    "Install it with `pip install numba` (or `pip install scoringrules[numba]`)."
)

# Set by ``backends.set_active``; read only by ``use_numba``.
_numba_active = False
_numba_installed_cache = None


def _set_numba_active(flag):
    global _numba_active
    _numba_active = bool(flag)


def _numba_installed():
    """Whether numba can be imported (checked once, on first use)."""
    global _numba_installed_cache
    if _numba_installed_cache is None:
        try:
            import numba  # noqa: F401
        except ImportError:
            _numba_installed_cache = False
        else:
            _numba_installed_cache = True
    return _numba_installed_cache


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
    is a numba request when numba is not installed (``ImportError``).
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
    if not _numba_installed():
        how = "backend='numba'" if explicit else "set_active('numba')"
        raise ImportError(
            f"{how} requires numba, which is not installed. " + NUMBA_INSTALL_HINT
        )
    return True
