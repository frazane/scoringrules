"""An array-API namespace augmented with the special functions and helpers
scoringrules needs that the standard does not provide.

The wrapper delegates every standard array-API op to the framework namespace
inferred from the input arrays (via array-api-compat) and adds the missing
special functions and a few non-standard helpers (see ``extensions``)."""

import numpy as np
from array_api_compat import array_namespace, is_array_api_obj

# Elementwise math functions that the core may call on Python-scalar constants
# (e.g. ``xp.sqrt(2.0)``, ``xp.log(2)``). The array-API standard requires array
# arguments and torch's namespace rejects bare Python floats, whereas the old
# backend methods coerced them. We restore that leniency by wrapping these so a
# Python-scalar first argument is converted to an array (a no-op for arrays).
_SCALAR_COERCE_FUNCS = frozenset(
    {
        "sqrt",
        "exp",
        "expm1",
        "log",
        "log1p",
        "log2",
        "log10",
        "abs",
        "floor",
        "ceil",
        "round",
        "trunc",
        "sign",
        "square",
        "reciprocal",
        "negative",
        "positive",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "tanh",
        "asinh",
        "acosh",
        "atanh",
        "isnan",
        "isinf",
        "isfinite",
    }
)


class ArrayAPINamespace:
    """A superset of an array-API namespace (bound to ``xp`` at call sites)."""

    def __init__(self, xp):
        self._xp = xp

    # --- standard ops: delegate everything else to the framework namespace ---
    def __getattr__(self, name):
        # Guard against infinite recursion if ``_xp`` is missing (e.g. a
        # partially-constructed instance): without it, ``self._xp`` would
        # re-enter __getattr__("_xp") forever.
        if name == "_xp":
            raise AttributeError("ArrayAPINamespace._xp is not set")
        attr = getattr(self._xp, name)
        if name in _SCALAR_COERCE_FUNCS and callable(attr):
            xp = self._xp

            def _scalar_tolerant(x, *args, **kwargs):
                if isinstance(x, (bool, int, float, complex)):
                    x = xp.asarray(x)
                return attr(x, *args, **kwargs)

            return _scalar_tolerant
        return attr

    # --- linear algebra (thin delegations so call sites stay mechanical) ---
    def norm(self, x, axis=None):
        # array-API standard spelling of the existing backends' linalg.norm
        # (equivalent for the L2/vector norm scoringrules uses).
        return self._xp.linalg.vector_norm(x, axis=axis)

    def inv(self, x):
        return self._xp.linalg.inv(x)

    def det(self, x):
        return self._xp.linalg.det(x)

    # --- non-standard helpers ---
    def gather(self, x, ind, axis):
        return self._xp.take_along_axis(x, ind, axis=axis)

    def apply_along_axis(self, func1d, x, axis):
        from . import extensions

        return extensions.apply_along_axis(self._xp, func1d, x, axis)

    def cov(self, x, rowvar=True, bias=False):
        from . import extensions

        return extensions.cov(self._xp, x, rowvar=rowvar, bias=bias)

    def indices(self, dimensions):
        from . import extensions

        return extensions.indices(self._xp, dimensions)

    # --- special functions (native-first, scipy fallback) ---
    def erf(self, x):
        from . import extensions

        return extensions.erf(self._xp, x)

    def gamma(self, x):
        from . import extensions

        return extensions.gamma(self._xp, x)

    def gammainc(self, x, y):
        from . import extensions

        return extensions.gammainc(self._xp, x, y)

    def gammalinc(self, x, y):
        from . import extensions

        return extensions.gammalinc(self._xp, x, y)

    def gammauinc(self, x, y):
        from . import extensions

        return extensions.gammauinc(self._xp, x, y)

    def beta(self, x, y):
        from . import extensions

        return extensions.beta(self._xp, x, y)

    def betainc(self, x, y, z):
        from . import extensions

        return extensions.betainc(self._xp, x, y, z)

    def mbessel0(self, x):
        from . import extensions

        return extensions.mbessel0(self._xp, x)

    def mbessel1(self, x):
        from . import extensions

        return extensions.mbessel1(self._xp, x)

    def hypergeometric(self, a, b, c, z):
        from . import extensions

        return extensions.hypergeometric(self._xp, a, b, c, z)

    def expi(self, x):
        from . import extensions

        return extensions.expi(self._xp, x)

    def comb(self, n, k):
        from . import extensions

        return extensions.comb(self._xp, n, k)

    def factorial(self, n):
        from . import extensions

        return extensions.factorial(self._xp, n)


_NUMPY_NS = ArrayAPINamespace(array_namespace(np.empty(0)))


def get_namespace(*arrays):
    """Return the augmented array-API namespace for ``arrays``.

    - all-scalar / list / ``None`` inputs (no recognised array) -> numpy;
    - arrays from one framework -> that framework's namespace;
    - arrays from different frameworks -> ``ValueError``.
    """
    candidates = [a for a in arrays if is_array_api_obj(a)]
    if not candidates:
        return _NUMPY_NS
    try:
        xp = array_namespace(*candidates)
    except TypeError as err:  # array-api-compat raises on multiple namespaces
        raise ValueError(
            "Inputs come from multiple array frameworks; convert all inputs to a "
            "single framework (e.g. all numpy, all jax, or all torch)."
        ) from err
    return ArrayAPINamespace(xp)
