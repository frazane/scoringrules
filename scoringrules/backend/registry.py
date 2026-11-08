"""Deprecated compatibility shim for the old backend registry.

``scoringrules.backends`` used to hold hand-written ``ArrayBackend`` objects.
Scores now infer the array framework from their inputs, so the registry only
remains for backward compatibility until 1.0: its values are the augmented
array-API namespaces (``ArrayAPINamespace``) of each framework. The numba
switch is ``scoringrules.set_config(numba=...)``; ``set_active`` maps to it."""

import importlib
import typing as tp
import warnings
from importlib.util import find_spec

from scoringrules import _dispatch
from scoringrules._config import set_config

from .namespace import _NUMPY_NS, ArrayAPINamespace, get_namespace

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Backend

_ALL_BACKENDS = ("jax", "numpy", "torch", "numba")
_REMOVAL_NOTE = (
    "is deprecated and removed in 1.0; scores infer the array framework from "
    "their inputs (use scoringrules.set_config(numba=...) to choose whether the "
    "numba gufuncs are used)."
)


def _make_namespace(name: str) -> ArrayAPINamespace:
    """The augmented namespace of ``name``, importing the framework on demand."""
    if name in {"numpy", "numba"}:
        return _NUMPY_NS
    if name == "jax":
        xp = importlib.import_module("jax.numpy")
    else:
        xp = importlib.import_module(name)
    return get_namespace(xp.zeros(1))


class BackendsRegistry(dict[str, ArrayAPINamespace]):
    """A dict-like container of registered backends (deprecated, removed in 1.0)."""

    def __init__(self):
        self.register_backend("numpy")
        # listed without importing numba (``import scoringrules`` must stay
        # cheap); the import happens on the first numba request
        if _dispatch._numba_available():
            super().__setitem__("numba", _NUMPY_NS)

        # Name returned by ``active``; the numba switch itself is the
        # ``numba`` option of ``scoringrules.set_config``.
        self._active = "numpy"

    @property
    def available_backends(self):
        """The backends whose library is installed.

        Neither of them is imported: jax, torch and numba are looked up with
        ``find_spec``. Once numba was imported (by the first numba request), a
        numba that failed to import is not listed.
        """
        return [
            b
            for b in _ALL_BACKENDS
            if (_dispatch._numba_available() if b == "numba" else find_spec(b))
        ]

    def register_backend(self, backend_name: "Backend"):
        """Register a backend. Currently supported backends are numpy, numba, jax, and torch."""
        if backend_name == "numba" and _dispatch._numba_import_error() is not None:
            raise _dispatch.numba_unavailable_error(
                "The numba backend", BackendNotAvailable
            )
        if backend_name not in self.available_backends:
            raise BackendNotAvailable(
                f"The backend '{backend_name}' is not available. "
                f"You need to install '{backend_name}'."
            )
        super().__setitem__(backend_name, _make_namespace(backend_name))

    def _get(self, key: str) -> ArrayAPINamespace:
        if key not in self:
            self.register_backend(key)
        return super().__getitem__(key)

    def __getitem__(self, __key: str) -> ArrayAPINamespace:
        """Get the array-API namespace of a backend (deprecated)."""
        warnings.warn(
            f"scoringrules.backends[{__key!r}] {_REMOVAL_NOTE}",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._get(__key)

    # The other accessors that hand out namespaces warn like item access. Key
    # lookups (``keys()``, ``in``, iteration) stay silent: they expose no
    # namespace and are what code checking for a backend uses.
    def get(self, key, default=None):
        """Like ``dict.get`` (deprecated, see ``__getitem__``)."""
        warnings.warn(
            f"scoringrules.backends.get({key!r}) {_REMOVAL_NOTE}",
            DeprecationWarning,
            stacklevel=2,
        )
        return super().get(key, default)

    def values(self):
        """Like ``dict.values`` (deprecated, see ``__getitem__``)."""
        warnings.warn(
            f"scoringrules.backends.values() {_REMOVAL_NOTE}",
            DeprecationWarning,
            stacklevel=2,
        )
        return super().values()

    def items(self):
        """Like ``dict.items`` (deprecated, see ``__getitem__``)."""
        warnings.warn(
            f"scoringrules.backends.items() {_REMOVAL_NOTE}",
            DeprecationWarning,
            stacklevel=2,
        )
        return super().items()

    def set_active(self, backend: str):
        """Select the global default backend (deprecated, removed in 1.0).

        ``"numba"`` is ``scoringrules.set_config(numba=True)`` (raises
        ``BackendNotAvailable`` if numba cannot be imported), ``"numpy"`` is
        ``scoringrules.set_config(numba=False)`` (plain numpy, as in 0.11), and
        any other name is ``scoringrules.set_config(numba="auto")``.
        """
        if backend == "numba":
            _dispatch.warn_at_caller(
                "set_active('numba') is deprecated and removed in 1.0; use "
                "scoringrules.set_config(numba=True) instead."
            )
            if _dispatch._numba_import_error() is not None:
                raise _dispatch.numba_unavailable_error(
                    "set_active('numba')", BackendNotAvailable
                )
            numba = True
        else:
            numba = False if backend == "numpy" else "auto"
            _dispatch.warn_at_caller(
                f"set_active({backend!r}) {_REMOVAL_NOTE} It now means "
                f"scoringrules.set_config(numba={numba!r})."
            )
        set_config(numba=numba)
        self._active = backend

    @property
    def active(self) -> ArrayAPINamespace:
        """The namespace of the backend last passed to ``set_active`` (deprecated)."""
        warnings.warn(
            f"scoringrules.backends.active {_REMOVAL_NOTE}",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._get(self._active)


class BackendNotAvailable(ImportError):
    """Raised if a backend library cannot be imported."""


class BackendNotRegistered(Exception):
    """Raised if a backend is not found in `BackendsRegistry`."""
