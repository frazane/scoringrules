from .registry import BackendsRegistry

backends = BackendsRegistry()

backends.register_backend("numpy")
if "numba" in backends.available_backends:
    backends.register_backend("numba")


def register_backend(backend_name: str):
    # global backends
    backends.register_backend(backend_name)


__all__ = ["backends"]
