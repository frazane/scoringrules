import typing as tp
from importlib.util import find_spec

from .base import ArrayBackend
from .jax import JaxBackend
from .numpy import NumbaBackend, NumpyBackend
from .tensorflow import TensorflowBackend
from .torch import TorchBackend

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Backend

_ALL_BACKENDS_MAP = {
    "jax": JaxBackend,
    "numpy": NumpyBackend,
    "torch": TorchBackend,
    "tensorflow": TensorflowBackend,
    "numba": NumbaBackend,
}

try:
    import numba  # noqa: F401

    _NUMBA_IMPORTED = True
except ImportError:
    _NUMBA_IMPORTED = False


class BackendsRegistry(dict[str, ArrayBackend]):
    """A dict-like container of registered backends."""

    def __init__(self):
        self.register_backend("numpy")
        if _NUMBA_IMPORTED:
            self.register_backend("numba")

        self._active = "numba" if _NUMBA_IMPORTED else "numpy"

    @property
    def available_backends(self):
        """Find if a backend library is available without importing it."""
        avail_backends = []
        for b in _ALL_BACKENDS_MAP:
            if find_spec(b) is not None:
                avail_backends.append(b)
        return avail_backends

    def register_backend(self, backend_name: "Backend"):
        """Register a backend. Currently supported backends are numpy, numba, jax, torch and tensorflow."""
        if backend_name not in self.available_backends:
            raise BackendNotAvailable(
                f"The backend '{backend_name}' is not available. "
                f"You need to install '{backend_name}'."
            )

        self[backend_name] = _ALL_BACKENDS_MAP[backend_name]()

    def __getitem__(self, __key: str) -> ArrayBackend:
        """Get a backend from the registry."""
        try:
            return super().__getitem__(__key)
        except KeyError:
            self.register_backend(__key)
            return super().__getitem__(__key)

    def set_active(self, backend: str):
        self._active = backend

    @property
    def active(self) -> ArrayBackend:
        return self[self._active]


class BackendNotAvailable(Exception):
    """Raised if a backend library cannot be imported."""


class BackendNotRegistered(Exception):
    """Raised if a backend is not found in `BackendsRegistry`."""
