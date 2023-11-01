from .registry import _NUMBA_IMPORTED, BackendsRegistry

backends = BackendsRegistry()


def register_backend(backend):
    backends.register_backend(backend)


__all__ = ["backends", "_NUMBA_IMPORTED", "BackendsRegistry", "register_backend"]
