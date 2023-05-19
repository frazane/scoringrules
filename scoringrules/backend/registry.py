from importlib.util import find_spec
from typing import Literal

from .base import AbstractBackend
from .numba import NumbaBackend
from .numpyapi import NumpyAPIBackend


class BackendsRegistry(dict[str, AbstractBackend]):
    """A dict-like container of registered backends."""

    @property
    def available_backends():
        """Find if an engine library is available without importing it."""
        avail_backends = ["numpy"]
        for b in ["numba", "jax"]:
            if find_spec(b) is not None:
                avail_backends.append(b)
        return avail_backends

    def register_engine(self, engine_name: Literal["numpy", "numba", "jax"]):
        """Register an engine. Currently supported backends are numpy, numba and jax."""
        if engine_name not in self.available_backends:
            raise BackendNotAvailable(
                f"The engine '{engine_name}' is not available. "
                f"You need to install '{engine_name}'."
            )

        if engine_name == "numpy":
            self["numpy"] = NumpyAPIBackend("numpy")
        elif engine_name == "jax":
            self["jax"] = NumpyAPIBackend("jax")
        elif engine_name == "numba":
            self["numba"] = NumbaBackend()

    def __getitem__(self, __key: str) -> AbstractBackend:
        """Get a backend from the registry."""
        try:
            return super().__getitem__(__key)
        except KeyError as err:
            raise BackendNotRegistered(
                f"The engine '{__key}' is not registered. "
                f"You can register it with scoringrules.register_backends('{__key}')"
            ) from err


class BackendNotAvailable(Exception):
    """Raised if a backend library cannot be imported."""


class BackendNotRegistered(Exception):
    """Raised if a backend is not found in `BackendsRegistry`."""
