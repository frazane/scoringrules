import warnings

from .namespace import ArrayAPINamespace, get_namespace
from .registry import BackendsRegistry

backends = BackendsRegistry()


def register_backend(backend):
    """Register a backend in ``scoringrules.backends`` (deprecated).

    The array framework is inferred from the inputs, so registering ``"numpy"``,
    ``"jax"`` or ``"torch"`` is deprecated and removed in 1.0. Registering
    ``"numba"`` checks that numba can be imported and registers the numpy
    namespace under ``"numba"``; it does not change whether the numba gufuncs
    are used, ``scoringrules.set_config(numba=...)`` does.

    Parameters
    ----------
    backend : str
        One of ``"numpy"``, ``"numba"``, ``"jax"`` or ``"torch"``.

    Raises
    ------
    BackendNotAvailable
        If the library is not installed (or, for numba, cannot be imported).
        It is a subclass of ``ImportError``.
    """
    if backend in {"numpy", "jax", "torch"}:
        warnings.warn(
            "register_backend for array-API backends is deprecated and removed "
            "in 1.0; the framework is inferred from the input.",
            DeprecationWarning,
            stacklevel=2,
        )
    backends.register_backend(backend)


__all__ = [
    "backends",
    "BackendsRegistry",
    "register_backend",
    "get_namespace",
    "ArrayAPINamespace",
]
