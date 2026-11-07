# scoringrules/_dispatch.py
"""Backend-argument handling for the array-API era.

``backend`` historically chose both the array library and numba. The library is
now inferred from the input, so the only meaningful values are ``None`` (infer)
and ``"numba"`` (gufunc fast-path). The legacy framework strings and the
registry remain accepted, with a DeprecationWarning, until 1.0."""

import warnings

from array_api_compat import is_array_api_obj

from scoringrules.backend import backends

_LEGACY_ARRAY_API_BACKENDS = {"numpy", "jax", "torch"}


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
    numpy-compatible. A numba request with non-numpy inputs is an error.
    """
    explicit = backend == "numba"
    global_numba = backend is None and backends._active == "numba"
    if not (explicit or global_numba):
        return False
    if not _is_numpy_compatible(*arrays):
        if explicit:
            raise ValueError(
                "backend='numba' requires numpy-compatible inputs; pass numpy "
                "arrays or drop backend='numba' to use the input's framework."
            )
        return False  # global numba default does not apply to jax/torch inputs
    return True
