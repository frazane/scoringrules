from importlib.util import find_spec
import typing as tp

import sys

from .base import ArrayBackend, Array
from .numpy import NumpyBackend
from .jax import JaxBackend
from .torch import TorchBackend

Backend = ArrayBackend[Array]

class BackendsRegistry(dict[str, Backend]):
    """A dict-like container of registered backends."""

    def __init__(self):
        self["numpy"] = NumpyBackend()

        if "jax" in sys.modules:
            self["jax"] = JaxBackend()
        
        if "torch" in sys.modules:
            self["torch"] = TorchBackend()

        self._active: str = "numpy"

    @property
    def available_backends(self):
        """Find if a backend library is available without importing it."""
        avail_backends = ["numpy"]
        for b in ["jax", "torch"]:
            if find_spec(b) is not None:
                avail_backends.append(b)
        return avail_backends

    def register_backend(self, backend_name: tp.Literal["jax", "torch"]):
        """Register a backend. Currently supported backends are numpy, numba and jax."""
        if backend_name not in self.available_backends:
            raise BackendNotAvailable(
                f"The backend '{backend_name}' is not available. "
                f"You need to install '{backend_name}'."
            )
        
        if backend_name == "jax":
            self["jax"] = JaxBackend()
        elif backend_name == "torch":
            self["torch"] = TorchBackend()

    def __getitem__(self, __key: str) -> Backend:

        """Get a backend from the registry."""
        try:
            return super().__getitem__(__key)
        except KeyError as err:
            raise BackendNotRegistered(
                f"The backend '{__key}' is not registered. "
                f"You can register it with scoringrules.register_backend('{__key}')"
            ) from err

    def set_active(self, backend: str):
        self._active = backend

    @property
    def active(self) -> Backend:
        return self[self._active]
    
class BackendNotAvailable(Exception):
    """Raised if a backend library cannot be imported."""


class BackendNotRegistered(Exception):
    """Raised if a backend is not found in `BackendsRegistry`."""
