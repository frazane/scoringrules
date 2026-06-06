from .namespace import ArrayAPINamespace, get_namespace
from .registry import BackendsRegistry

backends = BackendsRegistry()


def register_backend(backend):
    backends.register_backend(backend)


__all__ = [
    "backends",
    "BackendsRegistry",
    "register_backend",
    "get_namespace",
    "ArrayAPINamespace",
]
