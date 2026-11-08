"""Choice between the numba gufuncs and the array-API path of a score.

The array framework is inferred from the input. For numpy inputs, scores with
a numba gufunc use it according to the ``numba`` option of ``set_config`` /
``config_context`` (``scoringrules._config``): ``"auto"`` (default) whenever
numba is installed except for the entries of ``AUTO_EXCLUDED``, ``True``
always, ``False`` never. ``use_numba`` makes that choice for each call.

``backend`` historically chose both the array library and numba. Every value
is deprecated and removed in 1.0: ``backend="numba"`` still forces the gufuncs
for that call, the other values are ignored."""

import os
import warnings
from importlib.util import find_spec

import numpy as np
from array_api_compat import is_array_api_obj

from scoringrules import _config

NUMBA_INSTALL_HINT = (
    "Install it with `pip install numba` (or `pip install scoringrules[numba]`)."
)

# Deprecation warnings skip the frames of this package, so that they point at
# the user's call also when a score forwards ``backend`` to another score.
_PACKAGE_PREFIX = (os.path.dirname(os.path.abspath(__file__)) + os.sep,)

_UNSET = object()
_numba_import_cache = _UNSET
_broken_numba_warned = False

# Scores that ``numba="auto"`` sends to the array path even for numpy inputs
# the gufunc supports (``numba=True`` and ``backend="numba"`` still use the
# gufunc). Keys are what the score passes to ``use_numba``: its name, or
# ``(name, estimator)`` for the scores that take an estimator. Values are the
# float dtypes (native byte order) in which ``obs`` and ``fct`` are excluded,
# ``None`` for all.
AUTO_EXCLUDED = {
    # the gufunc's median term differs from the array path (#140, fix in #141)
    "weighted_interval_score": None,
    # numpy's float32 exp beats the gufunc on this O(M) work (0.4x)
    ("gksuv_ensemble", "akr"): frozenset({np.dtype(np.float32)}),
    ("gksuv_ensemble", "akr_circperm"): frozenset({np.dtype(np.float32)}),
}


class _ForwardedNumba:
    """``backend="numba"`` after ``resolve_backend_arg`` warned about it.

    A score that forwards ``backend`` to another score passes this instead,
    so that the deprecation warning is not raised a second time."""

    def __repr__(self):
        return "'numba'"


FORWARDED_NUMBA = _ForwardedNumba()


def _numba_import_error():
    """``None`` if numba imports, else the ``ImportError`` it raised (cached).

    This is the real import (about 0.2 s), done on the first numba request."""
    global _numba_import_cache
    if _numba_import_cache is _UNSET:
        try:
            import numba  # noqa: F401
        except ImportError as err:
            _numba_import_cache = err
            # the registry listed numba as installed at import: unlist it
            from scoringrules.backend import backends

            dict.pop(backends, "numba", None)
        else:
            _numba_import_cache = None
    return _numba_import_cache


def _numba_available():
    """Whether numba can be used, without importing it.

    Until the first numba request imports it (``_numba_import_error``), this is
    only a check that numba is installed; afterwards it is the import result."""
    if _numba_import_cache is _UNSET:
        try:
            return find_spec("numba") is not None
        except ValueError:  # numba in sys.modules without a spec
            return True
    return _numba_import_cache is None


def numba_unavailable_error(what, cls=ImportError):
    """The exception to raise when ``what`` needs numba but numba cannot be used.

    A missing numba gets the install hint; a numba that is installed but fails
    to import (e.g. a numba/numpy version mismatch) reports and chains the
    original error instead."""
    err = _numba_import_error()
    missing = isinstance(err, ModuleNotFoundError) and err.name == "numba"
    if missing:
        exc = cls(
            f"{what} requires numba, which is not installed. {NUMBA_INSTALL_HINT}"
        )
    else:
        exc = cls(f"{what} requires numba, but it could not be imported: {err}")
        exc.__cause__ = err
    exc.__suppress_context__ = True
    return exc


def warn_at_caller(message, category=DeprecationWarning):
    """Warn at the first caller outside scoringrules."""
    warnings.warn(message, category, skip_file_prefixes=_PACKAGE_PREFIX)


def resolve_backend_arg(backend):
    """Warn if ``backend`` was passed; return what a forwarding score passes on.

    Every value but ``None`` is deprecated. The return value is
    ``FORWARDED_NUMBA`` for ``backend="numba"`` (``use_numba`` treats it as
    ``"numba"`` without warning again) and ``None`` otherwise."""
    if backend is None:
        return None
    if backend is FORWARDED_NUMBA:
        return backend
    if backend == "numba":
        warn_at_caller(
            "Passing backend='numba' is deprecated and will be removed in 1.0. "
            "The numba gufuncs are used automatically for numpy inputs when numba "
            "is installed; use scoringrules.set_config(numba=True) or "
            "scoringrules.config_context(numba=True) to force them."
        )
        return FORWARDED_NUMBA
    warn_at_caller(
        f"Passing backend={backend!r} is deprecated and will be removed in 1.0. "
        "The array framework is inferred from the input; remove the backend "
        "argument, and use scoringrules.set_config(numba=...) or "
        "scoringrules.config_context(numba=...) to choose whether numba is used."
    )
    return None


def _is_numpy_compatible(*arrays):
    for a in arrays:
        if is_array_api_obj(a):
            name = type(a).__module__
            if not name.startswith("numpy"):
                return False
    return True


def gufunc_inputs_supported(*arrays):
    """Whether the numba gufuncs handle ``arrays`` exactly like the array path.

    True iff every entry is ``None``, a Python int or float, a list or tuple,
    an exact ``np.ndarray`` or ``np.memmap`` or a numpy scalar, and its numpy
    dtype (after ``np.asarray`` for the Python ones) is integer, float32 or
    float64, in any byte order (the gufunc wrapper casts to native order).
    Everything else takes the array path:

    - other ``np.ndarray`` subclasses (masked arrays, ``np.matrix``), whose
      semantics the gufuncs would drop, and objects that only implement
      ``__array__``;
    - bool data (arrays, Python and numpy bools): most array paths refuse it
      (``TypeError``), and the result must not depend on whether numba is
      installed. Flags such as ``bias`` are not passed here;
    - float16: the gufuncs compute and return float32, the array path float16;
    - long double, complex, object, string and datetime dtypes, and Python ints
      outside the int64/uint64 range, which numba has no loops for.

    Cheap for numpy input (a type and a dtype check per array); lists and
    tuples are converted once to find their dtype.
    """
    for a in arrays:
        t = type(a)
        if t is np.ndarray or t is np.memmap or isinstance(a, np.generic):
            dtype = a.dtype
        elif a is None or t is float:
            continue
        elif t is int or t is list or t is tuple:
            try:
                dtype = np.asarray(a).dtype
            except (TypeError, ValueError):  # e.g. ragged lists
                return False
        else:
            return False
        # "e" is float16, "g" long double (8 bytes on some platforms)
        if dtype.kind not in "iuf" or dtype.itemsize > 8 or dtype.char in "eg":
            return False
    return True


def _numba_usable_quietly():
    """``numba="auto"``: whether numba imports (imported on the first call).

    A missing numba is silent. A numba that is installed but fails to import
    warns once per process, since the user probably expects it to be used."""
    global _broken_numba_warned
    err = _numba_import_error()
    if err is None:
        return True
    missing = isinstance(err, ModuleNotFoundError) and err.name == "numba"
    if not missing and not _broken_numba_warned:
        _broken_numba_warned = True
        warn_at_caller(
            f"numba is installed but could not be imported ({err}); the scores "
            "use the array path. Use scoringrules.set_config(numba=False) to "
            "silence this warning.",
            RuntimeWarning,
        )
    return False


def _float_dtype(arrays):
    """The float dtype (native byte order) the gufunc computes ``arrays`` in."""
    dtypes = {getattr(a, "dtype", None) for a in arrays}
    if len(dtypes) == 1 and None not in dtypes:  # the common case, ~10x faster
        (dtype,) = dtypes
    else:
        dtype = np.result_type(
            *(np.asarray(a) if type(a) in (list, tuple) else a for a in arrays)
        )
    if dtype.kind != "f":
        return np.dtype(np.float64)
    return dtype if dtype.isnative else dtype.newbyteorder("=")


def _auto_excluded(key, arrays):
    dtypes = AUTO_EXCLUDED[key]
    return dtypes is None or _float_dtype(arrays) in dtypes


def use_numba(key, backend, *arrays, aux=()):
    """Whether this call of the score ``key`` uses its numba gufunc.

    ``key`` names the score (``"vs_ensemble"``), or the score and estimator
    (``("gksuv_ensemble", "akr")``) for the scores that take an estimator; it
    is looked up in ``AUTO_EXCLUDED``. ``arrays`` are ``obs`` and ``fct`` (or
    all the data inputs), ``aux`` the auxiliary inputs that reach the gufunc
    next to them (ensemble and dimension weights, ``alpha``, ``p``).

    The numba mode is ``True`` for ``backend="numba"`` (deprecated) and the
    ``numba`` option of the configuration otherwise. Then:

    - ``False``: no gufunc.
    - Inputs that are not numpy-compatible (jax, torch) take their framework's
      array path; with ``backend="numba"`` that is a ``ValueError``. Only
      ``arrays`` are checked: a jax or torch auxiliary input takes the array
      path through the input guard below.
    - Inputs the gufuncs do not support (``gufunc_inputs_supported``: masked
      arrays, float16, bool, ...) take the array path, also with ``True``.
    - ``True``: a call that would use a gufunc raises ``ImportError`` if numba
      cannot be imported.
    - ``"auto"``: no gufunc for the ``AUTO_EXCLUDED`` entries, or if numba is
      missing (silently; numba is imported by the first call that gets here).
    """
    explicit = backend is FORWARDED_NUMBA or backend == "numba"
    mode = True if explicit else _config._numba_mode()
    if mode is False:
        return False
    if not _is_numpy_compatible(*arrays):
        if explicit:
            raise ValueError(
                "backend='numba' requires numpy-compatible inputs; pass numpy "
                "arrays or drop backend='numba' to use the input's framework."
            )
        return False  # the numba option does not apply to jax/torch inputs
    if not gufunc_inputs_supported(*arrays, *aux):
        return False
    if mode is True:
        if _numba_import_error() is not None:
            how = "backend='numba'" if explicit else "numba=True"
            raise numba_unavailable_error(how)
        return True
    if key in AUTO_EXCLUDED and _auto_excluded(key, arrays):
        return False
    return _numba_usable_quietly()
