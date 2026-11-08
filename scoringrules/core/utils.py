"""Input checks and pre-processing shared by the scoring rules.

The array helpers take an ``xp`` augmented array-API namespace (see
``scoringrules.backend.get_namespace``). ``lazy_gufunc_wrapper_uv`` and
``lazy_gufunc_wrapper_mv`` wrap the numba gufuncs and work on numpy arrays;
``lazy_gufunc_getattr`` loads the gufunc modules of a family on first use. None
of them imports numba.
"""

import importlib
import sys
import threading

import numpy as np
from array_api_compat import array_namespace, is_array_api_obj

_V_AXIS = -1  # variable index for multivariate forecasts
_M_AXIS = -2  # ensemble index for multivariate forecasts
_M_AXIS_UV = -1  # ensemble index for univariate forecasts
_K_AXIS = -1  # categories index for categorical forecasts
EPSILON = 1e-5


def _shape_compatibility_check(obs, fct, m_axis) -> None:
    f_shape = fct.shape
    o_shape = obs.shape
    o_shape_broadcast = o_shape[:m_axis] + (f_shape[m_axis],) + o_shape[m_axis:]
    if o_shape_broadcast != f_shape:
        raise ValueError(
            f"Forecasts shape {f_shape} and observations shape {o_shape} are not compatible for broadcasting!"
        )


def estimator_check(estimator, gufuncs):
    """Check that the estimator is valid for the given score."""
    if estimator not in gufuncs:
        raise ValueError(
            f"{estimator} is not a valid estimator. Must be one of {tuple(gufuncs)}"
        )


def mv_weighted_score_chain(obs, fct, v_func):
    """Calculate transformed observations and ensemble members for threshold-weighted scoring rules given a chaining function."""
    obs, fct = map(v_func, (obs, fct))
    return obs, fct


def nan_policy_check(nan_policy: str) -> None:
    """Validate the nan_policy argument."""
    valid = ("propagate", "omit", "raise")
    if nan_policy not in valid:
        raise ValueError(f"Invalid nan_policy '{nan_policy}'. Must be one of {valid}.")


def _dtype_of(a):
    """The numpy dtype numba types ``a`` as: an array's dtype, float64 for a
    Python float, int64 for a Python int and bool for a Python bool."""
    try:
        return a.dtype
    except AttributeError:
        return np.asarray(a).dtype


def _gufunc_out_dtype(d0, d1):
    """Output dtype of a gufunc call on ``obs`` and ``fct`` of dtypes ``d0`` and
    ``d1``, as on the array path: numpy's promotion of the two, and numpy's
    default float64 when neither is floating (bool and integer inputs). At least
    float32, in native byte order. Auxiliary arguments (weights, exponents,
    constants built by the caller) do not take part."""
    dtype = np.result_type(d0, d1)
    if dtype.kind != "f":
        return np.dtype(np.float64)
    return np.result_type(dtype, np.float32).newbyteorder("=")


_FLOAT16 = np.dtype(np.float16)


def _native_dtype(dtype):
    """``dtype`` in native byte order, float16 as float32 (numba has neither)."""
    return (
        np.result_type(dtype, np.float32)
        if dtype.kind == "f"
        else dtype.newbyteorder("=")
    )


class GufuncWrapper:
    """Call a lazy (signature-less) numba gufunc and return its output.

    The output is allocated here, with the shape of ``obs`` (the first argument),
    reduced by its last axis for multivariate gufuncs, and the dtype of
    ``_gufunc_out_dtype``. ``obs`` and ``fct`` (the first two arguments) are cast
    to that dtype when theirs differs (integer, bool, float16, the other float
    precision, non-native byte order), so they always run the loop of the output
    dtype; the other arguments are passed as they are, except that non-native
    byte order and float16 (which numba does not support) become native and
    float32.

    numba's own ``GUFunc.__call__`` types every argument and searches the loops
    in Python on every call (about 50 us), and the search accepts safe casts, so
    float32 input would run through an existing float64 loop. Instead the dtypes
    of the call form a key: a known key calls the built numpy ufunc with exactly
    that loop (``signature=key``), an unknown one first compiles exactly that loop
    (loaded from numba's cache when ``cache=True`` and it is there). That uses
    numba internals; if they are missing, the wrapper calls the GUFunc instead.
    """

    def __init__(self, gufunc, mv):
        if not getattr(gufunc, "is_dynamic", False):
            raise TypeError(f"{gufunc!r} must be a lazy gufunc (no type signatures)")
        self._gufunc = gufunc
        self._mv = mv
        self._keys = set()
        self._out_dtypes = {}
        # numba builds a new ufunc per loop and assigns it after releasing its
        # compiler lock, so concurrent first calls must not interleave
        self._lock = threading.Lock()
        self._fallback = False  # numba internals missing: call the GUFunc
        self.__name__ = gufunc.__name__
        self.__doc__ = gufunc.__doc__
        self.__wrapped__ = gufunc

    def __repr__(self):
        return f"<GufuncWrapper {self.__name__}>"

    def _compile(self, args, key):
        """Compile the loop for exactly the dtypes of ``args`` and rebuild the
        ufunc. Returns False if numba's internals are not there (the wrapper then
        falls back to calling the GUFunc)."""
        g = self._gufunc
        with self._lock:
            if key in self._keys or self._fallback:
                return not self._fallback
            try:
                if g.find_ewise_function(g._get_ewise_dtypes(args)) == (None, None):
                    g.add(g._get_function_type(*args))
                g.build_ufunc()
            except (AttributeError, TypeError):
                self._fallback = True
                return False
            self._keys.add(key)
            return True

    def __call__(self, *args):
        a0, a1, *rest = args
        d0, d1 = _dtype_of(a0), _dtype_of(a1)
        odt = self._out_dtypes.get((d0, d1))
        if odt is None:
            odt = self._out_dtypes[(d0, d1)] = _gufunc_out_dtype(d0, d1)
        if d0 != odt:
            a0, d0 = np.asarray(a0, dtype=odt), odt
        if d1 != odt:
            a1, d1 = np.asarray(a1, dtype=odt), odt
        dts = [_dtype_of(a) for a in rest]
        for i, d in enumerate(dts):
            if not d.isnative or d == _FLOAT16:  # numba supports neither
                dts[i] = _native_dtype(d)
                rest[i] = np.asarray(rest[i], dtype=dts[i])
        shape = np.shape(a0)
        out = np.empty(shape[:-1] if self._mv else shape, dtype=odt)
        call = (a0, a1, *rest, out)
        key = (d0, d1, *dts, odt)
        if not self._fallback and (key in self._keys or self._compile(call, key)):
            try:
                self._gufunc.ufunc(*call, signature=key)
            except TypeError:
                # the loop was lost (e.g. numba rebuilt the ufunc from another
                # thread outside this wrapper): compile it again, once
                with self._lock:
                    self._keys.discard(key)
                if self._compile(call, key):
                    self._gufunc.ufunc(*call, signature=key)
                else:
                    self._gufunc(*call)
        else:
            self._gufunc(*call)
        if out.ndim == 0:
            return out[()]  # a numpy scalar of out's dtype
        return out


def lazy_gufunc_wrapper_uv(func):
    """Wrap a univariate lazy gufunc (output shaped like ``obs``), see ``GufuncWrapper``."""
    return GufuncWrapper(func, mv=False)


def lazy_gufunc_wrapper_mv(func):
    """Wrap a multivariate lazy gufunc (output shaped like ``obs`` without its last
    axis), see ``GufuncWrapper``."""
    return GufuncWrapper(func, mv=True)


def lazy_gufunc_getattr(package, attrs):
    """A module ``__getattr__`` for a score family that imports its numba gufunc
    modules on first access, so ``import scoringrules`` does not import numba.

    ``attrs`` maps each attribute name to the submodule that defines it, with
    an optional ``:name`` when it is defined under another name (e.g.
    ``{"estimator_gufuncs": "._gufuncs", "_ess": "._gufunc:_ess_gufunc"}``). The
    value is stored in the package, so later lookups are plain attribute reads.
    Without numba the attribute is ``None``; callers reach it only after
    ``use_numba`` checked numba.
    """

    def __getattr__(name):
        try:
            module, _, source = attrs[name].partition(":")
        except KeyError:
            raise AttributeError(
                f"module {package!r} has no attribute {name!r}"
            ) from None
        try:
            mod = importlib.import_module(module, package)
            value = getattr(mod, source or name)
        except ImportError:
            value = None
        except RuntimeError as err:
            if "no locator available" not in str(err):
                raise
            raise RuntimeError(
                "numba cannot cache the scoringrules gufuncs: neither the package "
                "directory nor numba's user cache directory is writable. Set the "
                "NUMBA_CACHE_DIR environment variable to a writable directory "
                f"before the first numba call. ({err})"
            ) from err
        setattr(sys.modules[package], name, value)
        return value

    return __getattr__


def univariate_array_check(obs, fct, m_axis, xp):
    obs, fct = xp.asarray(obs), xp.asarray(fct)
    m_axis = m_axis if m_axis >= 0 else fct.ndim + m_axis
    _shape_compatibility_check(obs, fct, m_axis)
    if m_axis != _M_AXIS_UV:
        fct = xp.moveaxis(fct, m_axis, _M_AXIS_UV)
    return obs, fct


def asarray_matching(*args, xp):
    """``xp.asarray`` each of ``args``, giving non-array ones (Python scalars,
    lists) the promoted float dtype and the device of the array ones.

    Plain ``xp.asarray`` would make a Python float float64 on numpy (upcasting
    float32 inputs) and float32 on torch (losing precision next to float64
    inputs). Arrays are returned unchanged and ``None`` is passed through. For
    weights that must follow one reference array, see ``asarray_like``.
    """
    arrays = [a for a in args if is_array_api_obj(a)]
    if not arrays:
        return tuple(None if a is None else xp.asarray(a) for a in args)
    dtype = xp.result_type(*(xp.float_dtype(a) for a in arrays))
    device = xp.device(arrays[0])
    return tuple(
        a
        if a is None or is_array_api_obj(a)
        else xp.asarray(a, dtype=dtype, device=device)
        for a in args
    )


def asarray_like(x, ref, *, xp):
    """``x`` as an ``xp`` array of ``xp.float_dtype(ref)`` on ``ref``'s device.

    The weights' form of ``asarray_matching`` (``ens_w``, the variogram ``w``):
    lists and scalars get ``ref``'s float dtype and device as there, and on top
    of that arrays of another framework or device (e.g. numpy weights for CUDA
    tensors) are converted and every result is cast to the float dtype, so the
    weights follow the forecast precision. Matching arrays only go through
    ``astype``: torch's ``asarray`` with keywords turns off ``requires_grad``.
    """
    foreign = is_array_api_obj(x) and (
        array_namespace(x) is not array_namespace(ref) or xp.device(x) != xp.device(ref)
    )
    if foreign:
        x = xp.asarray(x, device=xp.device(ref))
    else:
        x, _ = asarray_matching(x, ref, xp=xp)
    return xp.astype(x, xp.float_dtype(ref))


def binary_array_check(obs, fct, *, xp):
    """Check binary observations (0 or 1) and forecast probabilities (in [0, 1])."""
    obs, fct = asarray_matching(obs, fct, xp=xp)
    if xp.any(fct < 0.0) or xp.any(fct > 1.0 + EPSILON):
        raise ValueError("Forecasted probabilities must be within 0 and 1.")
    if not xp.all((obs == 0) | (obs == 1)):
        raise ValueError("Observations must be 0, 1, or NaN.")
    return obs, fct


def categorical_array_check(obs, fct, k_axis, onehot, *, xp):
    """Check categorical forecasts, move the category axis last and one-hot encode
    ``obs`` (categories ``1..K``) unless it already is."""
    obs, fct = asarray_matching(obs, fct, xp=xp)
    if xp.any(fct < 0.0) or xp.any(fct > 1.0 + EPSILON):
        raise ValueError("Forecasted probabilities must be within 0 and 1.")
    if k_axis != -1:
        fct = xp.moveaxis(fct, k_axis, _K_AXIS)
        if onehot:
            obs = xp.moveaxis(obs, k_axis, _K_AXIS)
    if onehot:
        if obs.shape != fct.shape:
            raise ValueError(
                f"Forecasts shape {fct.shape} and observations shape {obs.shape} are not compatible for broadcasting!"
            )
    else:
        if obs.shape != fct.shape[:-1]:
            raise ValueError(
                f"Forecasts shape {fct.shape} and observations shape {obs.shape} are not compatible for broadcasting!"
            )
        categories = xp.arange(1, fct.shape[-1] + 1, device=xp.device(obs))
        obs = xp.astype(xp.expand_dims(obs, axis=-1) == categories, xp.float_dtype(fct))
    return obs, fct


def _multivariate_shape_permute(obs, fct, m_axis, v_axis, *, xp):
    v_axis_obs = v_axis - 1 if m_axis < v_axis else v_axis
    fct = xp.moveaxis(fct, (m_axis, v_axis), (_M_AXIS, _V_AXIS))
    obs = xp.moveaxis(obs, v_axis_obs, _V_AXIS)
    return obs, fct


def multivariate_array_check(obs, fct, m_axis, v_axis, *, xp):
    """Check and adapt the shapes of multivariate forecasts and observations arrays."""
    obs, fct = asarray_matching(obs, fct, xp=xp)
    m_axis = m_axis if m_axis >= 0 else fct.ndim + m_axis
    v_axis = v_axis if v_axis >= 0 else fct.ndim + v_axis
    _shape_compatibility_check(obs, fct, m_axis)
    return _multivariate_shape_permute(obs, fct, m_axis, v_axis, xp=xp)


def univariate_weight_check(ens_w, fct, m_axis, xp):
    if ens_w is not None:
        # float_dtype, not fct.dtype: integer forecasts keep fractional weights
        ens_w = asarray_like(ens_w, fct, xp=xp)
        m_axis = m_axis if m_axis >= 0 else fct.ndim + m_axis
        ens_w = xp.moveaxis(ens_w, m_axis, _M_AXIS_UV)
        if ens_w.shape != fct.shape:
            raise ValueError(
                f"Shape of weights {ens_w.shape} is not compatible with forecast "
                f"shape {fct.shape}"
            )
        if xp.any(ens_w < 0):
            raise ValueError("`ens_w` contains negative entries")
        ens_w = ens_w / xp.sum(ens_w, axis=-1, keepdims=True)
    return ens_w


def univariate_sort_ens(
    fct, ens_w=None, m_axis=-1, estimator=None, sorted_ensemble=False, *, xp
):
    sort_ensemble = not sorted_ensemble and estimator in ["qd", "pwm", "int"]
    # stable=False: the array-API default (stable=True) selects numpy's much
    # slower stable sort. The order of tied members only changes rounding:
    # equal members have equal values, and the weighted estimators are
    # symmetric in tied members.
    if sort_ensemble and ens_w is None:
        fct = xp.sort(fct, axis=-1, stable=False)
    elif sort_ensemble:
        ind = xp.argsort(fct, axis=-1, stable=False)
        fct = xp.gather(fct, ind, axis=-1)
    if ens_w is not None:
        ens_w = univariate_weight_check(ens_w, fct, m_axis, xp=xp)
        if sort_ensemble:
            ens_w = xp.gather(ens_w, ind, axis=-1)
    return fct, ens_w


def uv_weighted_score_weights(
    obs, fct, a=float("-inf"), b=float("inf"), w_func=None, *, xp
):
    if w_func is None:

        def w_func(x):
            # astype, not ``* 1.0``: bool * float defaults to float64 (numpy) /
            # float32 (torch) instead of following the input's precision.
            return xp.astype((a <= x) & (x <= b), xp.float_dtype(x))

    obs_w, fct_w = map(w_func, (obs, fct))
    obs_w, fct_w = xp.asarray(obs_w), xp.asarray(fct_w)
    if xp.any(obs_w < 0) or xp.any(fct_w < 0):
        raise ValueError("`w_func` returns negative values")
    return obs_w, fct_w


def uv_outcome_weights(obs, fct, a, b, w_func, *, xp):
    """``uv_weighted_score_weights`` cast to ``xp.float_dtype(fct)`` (with
    ``asarray_like``), so boolean, integer or other-precision weights follow the
    forecast precision on both paths."""
    obs_w, fct_w = uv_weighted_score_weights(obs, fct, a, b, w_func, xp=xp)
    return asarray_like(obs_w, fct, xp=xp), asarray_like(fct_w, fct, xp=xp)


def uv_weighted_score_chain(
    obs, fct, a=float("-inf"), b=float("inf"), v_func=None, *, xp
):
    if v_func is None:
        a, b, obs, fct = (
            xp.asarray(a),
            xp.asarray(b),
            xp.asarray(obs),
            xp.asarray(fct),
        )

        def v_func(x):
            return xp.minimum(xp.maximum(x, a), b)

    obs, fct = map(v_func, (obs, fct))
    return obs, fct


def apply_nan_policy_ens_uv(
    obs, fct, nan_policy="propagate", ens_w=None, estimator=None, m_axis=-1, *, xp
):
    if ens_w is not None:
        ens_w = xp.moveaxis(xp.asarray(ens_w, dtype=xp.float_dtype(fct)), m_axis, -1)

    if nan_policy == "propagate":
        return obs, fct, ens_w

    nan_mask = xp.isnan(fct)
    if ens_w is not None:
        nan_mask = nan_mask | xp.isnan(ens_w)

    if nan_policy == "raise":
        if xp.any(nan_mask) or xp.any(xp.isnan(obs)):
            raise ValueError(
                "NaN values encountered in input. Use nan_policy='propagate' or "
                "nan_policy='omit' to handle NaN values."
            )
        return obs, fct, ens_w

    if nan_policy == "omit":
        if estimator in ["int", "akr", "akr_circperm"]:
            raise NotImplementedError(
                f"NaN handling with nan_policy='omit' is not implemented for "
                f"estimator '{estimator}'."
            )
        fct = xp.where(xp.isnan(fct), 0.0, fct)
        if ens_w is None:
            ens_w = xp.asarray(~nan_mask, dtype=xp.float_dtype(fct))
        else:
            ens_w = xp.where(nan_mask, 0.0, ens_w)
        return obs, fct, ens_w

    raise ValueError(
        f"Invalid nan_policy '{nan_policy}'. Must be one of 'propagate', 'omit', "
        "'raise'."
    )


def _mv_weights_ensemble_last(ens_w, fct, m_axis, v_axis, *, xp):
    """``ens_w`` as ``xp.float_dtype(fct)`` with its ensemble axis last.

    ``m_axis`` and ``v_axis`` describe the forecast layout the weights follow:
    ``ens_w`` is that layout without the variables axis, so its ensemble axis
    shifts down by one when ``v_axis`` comes before ``m_axis``. ``fct`` is
    only used for its rank, dtype and device.
    """
    ens_w = asarray_like(ens_w, fct, xp=xp)
    if ens_w.ndim != fct.ndim - 1:
        raise ValueError(
            f"Shape of weights {ens_w.shape} is not compatible with forecast "
            f"shape {fct.shape}"
        )
    m_axis = m_axis if m_axis >= 0 else fct.ndim + m_axis
    v_axis = v_axis if v_axis >= 0 else fct.ndim + v_axis
    w_axis = m_axis - 1 if v_axis < m_axis else m_axis
    return xp.moveaxis(ens_w, w_axis, -1)


def multivariate_weight_check(ens_w, fct, m_axis, v_axis, *, xp):
    """Check ensemble weights against the permuted ``fct`` (``..., M, D``).

    ``m_axis`` and ``v_axis`` give the layout ``ens_w`` follows (see
    ``_mv_weights_ensemble_last``): the caller's axes for weights as passed by
    the user, or ``-2, -1`` for weights already moved ensemble-last. Returns
    the weights ensemble-last, as ``xp.float_dtype(fct)``, normalised to sum
    to one over the members.
    """
    if ens_w is None:
        return None
    ens_w = _mv_weights_ensemble_last(ens_w, fct, m_axis, v_axis, xp=xp)
    if ens_w.shape != fct.shape[:-1]:
        raise ValueError(
            f"Shape of weights {ens_w.shape} is not compatible with forecast "
            f"shape {fct.shape}"
        )
    if xp.any(ens_w < 0):
        raise ValueError("`ens_w` contains negative entries")
    return ens_w / xp.sum(ens_w, axis=-1, keepdims=True)


def mv_weighted_score_weights(obs, fct, w_func, *, xp):
    """Outcome weights ``w_func(obs)`` (``...``) and ``w_func(fct)`` (``..., M``).

    ``w_func`` receives one vector of the ``D`` variables at a time and returns
    a scalar. The weights are cast to ``xp.float_dtype(fct)``, so boolean or
    Python-float results follow the forecast precision.
    """
    dtype = xp.float_dtype(fct)
    obs_w = xp.astype(xp.asarray(xp.apply_along_axis(w_func, obs, -1)), dtype)
    fct_w = xp.astype(xp.asarray(xp.apply_along_axis(w_func, fct, -1)), dtype)
    if xp.any(obs_w < 0) or xp.any(fct_w < 0):
        raise ValueError("`w_func` returns negative values")
    return obs_w, fct_w


def apply_nan_policy_ens_mv(
    obs,
    fct,
    nan_policy="propagate",
    ens_w=None,
    estimator=None,
    m_axis=-2,
    v_axis=-1,
    *,
    xp,
):
    """Apply a NaN policy to permuted multivariate forecasts (``..., M, D``).

    A member is invalid if any of its variables, or its weight, is NaN. NaN in
    ``obs`` always propagates. A user ``ens_w`` (laid out like the original
    ``fct`` given ``m_axis``/``v_axis``, without the variables axis) is returned
    ensemble-last as ``xp.float_dtype(fct)``, whatever the policy.

    - ``'propagate'``: no-op.
    - ``'raise'``: ``ValueError`` if ``obs``, ``fct`` or ``ens_w`` has a NaN.
    - ``'omit'``: invalid members get zero weight (a 0/1 mask when no weights
      were given) and their NaN values become ``0.0``, so the caller routes the
      score through the weighted path. Not implemented for ``'akr'`` and
      ``'akr_circperm'``.
    """
    if ens_w is not None:
        ens_w = _mv_weights_ensemble_last(ens_w, fct, m_axis, v_axis, xp=xp)

    if nan_policy == "propagate":
        return obs, fct, ens_w

    nan_mask = xp.any(xp.isnan(fct), axis=-1)  # (..., M)
    if ens_w is not None:
        nan_mask = nan_mask | xp.isnan(ens_w)

    if nan_policy == "raise":
        if xp.any(nan_mask) or xp.any(xp.isnan(obs)):
            raise ValueError(
                "NaN values encountered in input. Use nan_policy='propagate' or "
                "nan_policy='omit' to handle NaN values."
            )
        return obs, fct, ens_w

    if nan_policy == "omit":
        if estimator in ["akr", "akr_circperm"]:
            raise NotImplementedError(
                f"NaN handling with nan_policy='omit' is not implemented for "
                f"estimator '{estimator}'."
            )
        # zero-weighted members never contribute; zeroing their values only
        # keeps NaN out of the arithmetic.
        fct = xp.where(nan_mask[..., None], 0.0, fct)
        if ens_w is None:
            ens_w = xp.astype(~nan_mask, xp.float_dtype(fct))
        else:
            ens_w = xp.where(nan_mask, 0.0, ens_w)
        return obs, fct, ens_w

    raise ValueError(
        f"Invalid nan_policy '{nan_policy}'. Must be one of 'propagate', 'omit', "
        "'raise'."
    )
