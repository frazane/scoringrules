from .registry import BackendsRegistry

backends = BackendsRegistry()

backends.register_engine("numpy")
if "numba" in backends.available_engines:
    backends.register_engine("numba")

__all__ = ["backends"]
