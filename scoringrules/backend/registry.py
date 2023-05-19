from importlib.util import find_spec
from typing import Literal

from .base import AbstractBackend
from .numba import NumbaBackend
from .numpyapi import NumpyAPIBackend


class BackendsRegistry(dict[str, AbstractBackend]):
    """A dict-like container of registered backends."""

    @property
    def available_backends(self):
        """Find if a backend library is available without importing it."""
        avail_backends = ["numpy"]
        for b in ["numba", "jax"]:
            if find_spec(b) is not None:
                avail_backends.append(b)
        return avail_backends

    def register_backend(self, backend_name: Literal["numpy", "numba", "jax"]):
        """Register a backend. Currently supported backends are numpy, numba and jax."""
        if backend_name not in self.available_backends:
            raise BackendNotAvailable(
                f"The backend '{backend_name}' is not available. "
                f"You need to install '{backend_name}'."
            )

        if backend_name == "numpy":
            self["numpy"] = NumpyAPIBackend("numpy")
        elif backend_name == "jax":
            self["jax"] = NumpyAPIBackend("jax")
        elif backend_name == "numba":
            self["numba"] = NumbaBackend()

    def __getitem__(self, __key: str) -> AbstractBackend:
        """Get a backend from the registry."""
        try:
            return super().__getitem__(__key)
        except KeyError as err:
            raise BackendNotRegistered(
                f"The backend '{__key}' is not registered. "
                f"You can register it with scoringrules.register_backend('{__key}')"
            ) from err


class BackendNotAvailable(Exception):
    """Raised if a backend library cannot be imported."""


class BackendNotRegistered(Exception):
    """Raised if a backend is not found in `BackendsRegistry`."""
