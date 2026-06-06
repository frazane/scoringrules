"""Special functions and non-standard helpers missing from the array-API
standard. Each takes the (compat) namespace ``xp`` and dispatches native-first,
falling back to scipy only where no native implementation exists.

The forward/gradient support matrix is documented in
``docs/special_functions_audit.md``."""

import numpy as np
import scipy.special as sp

# NOTE: the scipy branches below only ever receive numpy arrays (jax/torch
# dispatch natively or raise), so SCIPY_ARRAY_API is irrelevant here.


def _kind(xp):
    name = getattr(xp, "__name__", "")
    if "torch" in name:
        return "torch"
    if "jax" in name:
        return "jax"
    return "numpy"


def _jsp():
    import jax.scipy.special as jsp

    return jsp


def _ts():
    import torch

    return torch


# --- special functions ---


def erf(xp, x):
    k = _kind(xp)
    if k == "jax":
        return _jsp().erf(x)
    if k == "torch":
        return _ts().special.erf(x)
    return sp.erf(x)


def gamma(xp, x):
    k = _kind(xp)
    if k == "jax":
        return _jsp().gamma(x)
    if k == "torch":
        return _ts().exp(_ts().lgamma(x))
    return sp.gamma(x)


def gammainc(xp, x, y):
    k = _kind(xp)
    if k == "jax":
        return _jsp().gammainc(x, y)
    if k == "torch":
        return _ts().special.gammainc(x, y)
    return sp.gammainc(x, y)


def gammalinc(xp, x, y):
    """Lower incomplete gamma (unregularised)."""
    k = _kind(xp)
    if k == "jax":
        return _jsp().gammainc(x, y) * _jsp().gamma(x)
    if k == "torch":
        return _ts().special.gammainc(x, y) * _ts().exp(_ts().lgamma(x))
    return sp.gammainc(x, y) * sp.gamma(x)


def gammauinc(xp, x, y):
    """Upper incomplete gamma (unregularised)."""
    k = _kind(xp)
    if k == "jax":
        return _jsp().gammaincc(x, y) * _jsp().gamma(x)
    if k == "torch":
        return _ts().special.gammaincc(x, y) * _ts().exp(_ts().lgamma(x))
    return sp.gammaincc(x, y) * sp.gamma(x)


def beta(xp, x, y):
    k = _kind(xp)
    if k == "jax":
        return _jsp().beta(x, y)
    if k == "torch":
        t = _ts()
        return t.exp(t.lgamma(x) + t.lgamma(y) - t.lgamma(x + y))
    return sp.beta(x, y)


def betainc(xp, x, y, z):
    k = _kind(xp)
    if k == "jax":
        return _jsp().betainc(x, y, z)
    if k == "torch":
        raise NotImplementedError(
            "betainc has no native torch implementation and the scipy fallback "
            "does not support autograd; pass numpy/jax arrays for this score."
        )
    return sp.betainc(x, y, z)


def mbessel0(xp, x):
    k = _kind(xp)
    if k == "jax":
        return _jsp().i0(x)
    if k == "torch":
        return _ts().special.i0(x)
    return sp.i0(x)


def mbessel1(xp, x):
    k = _kind(xp)
    if k == "jax":
        return _jsp().i1(x)
    if k == "torch":
        return _ts().special.i1(x)
    return sp.i1(x)


def hypergeometric(xp, a, b, c, z):
    k = _kind(xp)
    if k == "jax":
        return _jsp().hyp2f1(a, b, c, z)
    if k == "torch":
        raise NotImplementedError(
            "hyp2f1 has no native torch implementation; pass numpy/jax arrays."
        )
    return sp.hyp2f1(a, b, c, z)


def expi(xp, x):
    k = _kind(xp)
    if k == "jax":
        return _jsp().expi(x)
    if k == "torch":
        raise NotImplementedError(
            "expi has no native torch implementation; pass numpy/jax arrays."
        )
    return sp.expi(x)


def factorial(xp, n):
    k = _kind(xp)
    if k == "jax":
        return _jsp().factorial(n)
    if k == "torch":
        t = _ts()
        return t.exp(t.lgamma(n + 1))
    return sp.factorial(n)


def comb(xp, n, k):
    # Compose from factorial. Use floating division + round rather than floor
    # division: jax's factorial is float32, and ``120.0001 // 12.00001`` floors
    # to 9.0 instead of 10.0. Rounding the exact-in-float ratio is robust on
    # numpy/jax/torch. comb is integer-valued, so this is not differentiable
    # (matching the old floor-division form).
    ratio = factorial(xp, n) / (factorial(xp, k) * factorial(xp, n - k))
    return xp.round(ratio)


# --- non-standard helpers ---


def apply_along_axis(xp, func1d, x, axis):
    k = _kind(xp)
    if k == "jax":
        import jax
        import jax.numpy as jnp

        try:
            # vmap fast path (scalar-returning func1d): move the target axis last
            # so the reshape groups slices correctly for ANY axis, not just -1.
            xm = jnp.moveaxis(x, axis, -1)
            shape = list(xm.shape)
            n = shape.pop()
            return jax.vmap(func1d)(xm.reshape(-1, n)).reshape(shape)
        except Exception:
            return jnp.apply_along_axis(func1d, axis, x)
    if k == "numpy":
        return np.apply_along_axis(func1d, axis, x)
    t = _ts()
    return t.as_tensor(
        np.apply_along_axis(
            lambda a: np.asarray(func1d(t.as_tensor(a))), axis, np.asarray(x)
        )
    )


def cov(xp, x, rowvar=True, bias=False):
    k = _kind(xp)
    if k == "jax":
        import jax.numpy as jnp

        return jnp.cov(x, rowvar=rowvar, bias=bias)
    if k == "torch":
        t = _ts()
        if not rowvar:
            x = x.T
        return t.cov(x, correction=0 if bias else 1)
    return np.cov(x, rowvar=rowvar, bias=bias)


def indices(xp, dimensions):
    k = _kind(xp)
    if k == "jax":
        import jax.numpy as jnp

        return jnp.indices(dimensions)
    if k == "torch":
        return _ts().as_tensor(np.indices(dimensions))
    return np.indices(dimensions)
