import warnings

from .namespace import ArrayAPINamespace, get_namespace
from .registry import BackendsRegistry

backends = BackendsRegistry()


def register_backend(backend):
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
