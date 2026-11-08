"""Special functions and non-standard helpers missing from the array-API
standard. Each takes the (compat) namespace ``xp`` and dispatches native-first,
falling back to scipy only where no native implementation exists.

The forward/gradient support matrix is documented in
``docs/special_functions_audit.md``."""

import math

import numpy as np
import scipy.special as sp
from array_api_compat import (
    is_array_api_obj,
    is_jax_namespace,
    is_numpy_namespace,
    is_torch_namespace,
)

# NOTE: the scipy branches below only ever receive numpy arrays (jax/torch
# dispatch natively or raise), so SCIPY_ARRAY_API is irrelevant here.


def _kind(xp):
    if is_numpy_namespace(xp):
        return "numpy"
    if is_jax_namespace(xp):
        return "jax"
    if is_torch_namespace(xp):
        return "torch"
    raise NotImplementedError(
        f"Array namespace {getattr(xp, '__name__', xp)!r} is not supported; "
        "scoringrules supports numpy, jax and torch arrays."
    )


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
    # sp.factorial always returns float64; keep float32 inputs in float32.
    out = sp.factorial(n)
    return out.astype(np.result_type(np.asarray(n).dtype, np.float32), copy=False)


def comb(xp, n, k):
    """Binomial coefficient for real ``n`` and ``k``, as ``scipy.special.comb``
    with ``exact=False``: 0 outside ``0 <= k <= n``, not rounded.

    numpy calls scipy, as in earlier releases. jax and torch work in log space,
    ``exp(lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1))``, so large ``n``
    neither overflows (factorials overflow float32 from 35! and float64 from
    171!) nor loses the float32 dtype, and the result is differentiable.
    """
    kind = _kind(xp)
    if kind == "numpy":
        out = sp.comb(n, k)
        dtype = np.result_type(np.asarray(n).dtype, np.asarray(k).dtype, np.float32)
        return out.astype(dtype, copy=False)
    lgamma = _jsp().gammaln if kind == "jax" else _ts().lgamma
    # Coerce scalars only: the compat ``torch.asarray`` drops ``requires_grad``.
    n = n if is_array_api_obj(n) else xp.asarray(n)
    k = k if is_array_api_obj(k) else xp.asarray(k)
    valid = (k >= 0) & (k <= n)
    # Evaluate the masked entries at n = k = 0 so they give a finite value
    # (and gradient) before being set to 0.
    safe_n = xp.where(valid, n, 0.0)
    safe_k = xp.where(valid, k, 0.0)
    log_comb = lgamma(safe_n + 1) - lgamma(safe_k + 1) - lgamma(safe_n - safe_k + 1)
    return xp.where(valid, xp.exp(log_comb), 0.0)


# --- non-standard helpers ---


def quantile(xp, x, q, axis=None, keepdims=False):
    """``numpy.quantile`` (linear interpolation) for every framework.

    ``q`` is a scalar or 1-d array of levels in ``[0, 1]``; for a 1-d ``q`` the
    levels form the leading axis of the result, as in numpy. ``axis`` is an int
    or ``None`` (flatten).
    """
    k = _kind(xp)
    if k == "jax":
        import jax.numpy as jnp

        return jnp.quantile(x, jnp.asarray(q), axis=axis, keepdims=keepdims)
    if k == "torch":
        t = _ts()
        # torch.quantile wants q as a tensor of x's dtype and device.
        q = t.as_tensor(q, dtype=x.dtype, device=x.device)
        return t.quantile(x, q, dim=axis, keepdim=keepdims)
    if xp.isdtype(x.dtype, "real floating"):
        # an array q would otherwise promote float32 x to float64
        q = np.asarray(q, dtype=x.dtype)
    return np.quantile(x, q, axis=axis, keepdims=keepdims)


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
    return _torch_apply_along_axis(func1d, x, axis)


def _torch_apply_along_axis(func1d, x, axis):
    """Torch-native ``apply_along_axis``: no numpy round-trip, so autograd and
    the input's device are preserved.

    Tries ``torch.vmap`` first; falls back to a Python loop over the 1-d slices
    for callables vmap cannot trace (data-dependent control flow, ``.item()``).
    Like ``numpy.apply_along_axis``, any dimensions ``func1d`` returns take the
    place of ``axis`` in the output.
    """
    t = _ts()
    if not -x.ndim <= axis < x.ndim:
        raise np.exceptions.AxisError(axis, x.ndim)
    axis = axis % x.ndim
    xm = t.movedim(x, axis, -1)
    batch_shape = tuple(xm.shape[:-1])
    # explicit batch size (not -1) so an empty ``axis`` still reshapes
    flat = xm.reshape(math.prod(batch_shape), xm.shape[-1])

    def as_row_tensor(r):
        # Python floats follow a floating input's precision; as_tensor alone
        # would make them float32 (torch's default dtype).
        if isinstance(r, float) and x.is_floating_point():
            return t.as_tensor(r, dtype=x.dtype, device=x.device)
        return t.as_tensor(r, device=x.device)

    try:
        out = t.vmap(func1d)(flat)
    except Exception:
        out = t.stack([as_row_tensor(func1d(row)) for row in t.unbind(flat)])
    tail = tuple(out.shape[1:])
    out = out.reshape(batch_shape + tail)
    if tail:
        nb = len(batch_shape)
        out = t.movedim(
            out,
            tuple(range(nb, nb + len(tail))),
            tuple(range(axis, axis + len(tail))),
        )
    return out
