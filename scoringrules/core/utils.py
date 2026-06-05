import typing as tp

import numpy as np

from scoringrules.backend import backends

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, Backend, NanPolicy

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


def binary_array_check(obs, fct, backend=None):
    """Check and adapt the shapes of binary forecasts and observations arrays."""
    B = backends.active if backend is None else backends[backend]
    obs, fct = map(B.asarray, (obs, fct))
    if B.any(fct < 0.0) or B.any(fct > 1.0 + EPSILON):
        raise ValueError("Forecasted probabilities must be within 0 and 1.")
    if not set(v.item() for v in B.unique_values(obs)) <= {0, 1}:
        raise ValueError("Observations must be 0, 1, or NaN.")
    return obs, fct


def categorical_array_check(obs, fct, k_axis, onehot, backend=None):
    """Check and adapt the shapes of categorical forecasts and observations arrays."""
    B = backends.active if backend is None else backends[backend]
    obs, fct = map(B.asarray, (obs, fct))
    if B.any(fct < 0.0) or B.any(fct > 1.0 + EPSILON):
        raise ValueError("Forecasted probabilities must be within 0 and 1.")
    if k_axis != -1:
        fct = B.moveaxis(fct, k_axis, _K_AXIS)
        if onehot:
            obs = B.moveaxis(obs, k_axis, _K_AXIS)
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
        categories = B.arange(1, fct.shape[-1] + 1)
        obs = B.where(B.expand_dims(obs, -1) == categories, 1, 0)
    return obs, fct


def univariate_array_check(obs, fct, m_axis, backend=None):
    """Check and adapt the shapes of univariate forecasts and observations arrays."""
    B = backends.active if backend is None else backends[backend]
    obs, fct = map(B.asarray, (obs, fct))
    m_axis = m_axis if m_axis >= 0 else fct.ndim + m_axis
    _shape_compatibility_check(obs, fct, m_axis)
    if m_axis != _M_AXIS_UV:
        fct = B.moveaxis(fct, m_axis, _M_AXIS_UV)
    return obs, fct


def _multivariate_shape_permute(obs, fct, m_axis, v_axis, backend=None):
    B = backends.active if backend is None else backends[backend]
    v_axis_obs = v_axis - 1 if m_axis < v_axis else v_axis
    fct = B.moveaxis(fct, (m_axis, v_axis), (_M_AXIS, _V_AXIS))
    obs = B.moveaxis(obs, v_axis_obs, _V_AXIS)
    return obs, fct


def multivariate_array_check(obs, fct, m_axis, v_axis, backend=None):
    """Check and adapt the shapes of multivariate forecasts and observations arrays."""
    B = backends.active if backend is None else backends[backend]
    obs, fct = map(B.asarray, (obs, fct))
    m_axis = m_axis if m_axis >= 0 else fct.ndim + m_axis
    v_axis = v_axis if v_axis >= 0 else fct.ndim + v_axis
    _shape_compatibility_check(obs, fct, m_axis)
    return _multivariate_shape_permute(obs, fct, m_axis, v_axis, backend=backend)


def univariate_weight_check(ens_w, fct, m_axis, backend=None):
    """Check that ensemble weights are non-negative and compatible with the (permuted) forecast array."""
    B = backends.active if backend is None else backends[backend]
    if ens_w is not None:
        ens_w = B.asarray(ens_w)
        m_axis = m_axis if m_axis >= 0 else fct.ndim + m_axis
        ens_w = B.moveaxis(ens_w, m_axis, _M_AXIS_UV)
        if ens_w.shape != fct.shape:
            raise ValueError(
                f"Shape of weights {ens_w.shape} is not compatible with forecast shape {fct.shape}"
            )
        if B.any(ens_w < 0):
            raise ValueError("`ens_w` contains negative entries")
        # renormalise weights so that they sum to one across the ensemble members
        ens_w = ens_w / B.sum(ens_w, axis=-1, keepdims=True)
    return ens_w


def multivariate_weight_check(ens_w, fct, m_axis, backend=None):
    """Check that ensemble weights are non-negative and compatible with the (permuted) forecast array."""
    B = backends.active if backend is None else backends[backend]
    if ens_w is not None:
        ens_w = B.asarray(ens_w)
        m_axis = m_axis if m_axis >= 0 else fct.ndim + m_axis
        v_axis_pos = _V_AXIS if _V_AXIS > 0 else _V_AXIS + fct.ndim
        ens_w = B.moveaxis(ens_w, m_axis, -1)
        if ens_w.shape != fct.shape[:v_axis_pos] + fct.shape[(v_axis_pos + 1) :]:
            raise ValueError(
                f"Shape of weights {ens_w.shape} is not compatible with forecast shape {fct.shape}"
            )
        if B.any(ens_w < 0):
            raise ValueError("`ens_w` contains negative entries")
        # renormalise weights so that they sum to one across the ensemble members
        ens_w = ens_w / B.sum(ens_w, axis=-1, keepdims=True)
    return ens_w


def estimator_check(estimator, gufuncs):
    """Check that the estimator is valid for the given score."""
    if estimator not in gufuncs:
        raise ValueError(
            f"{estimator} is not a valid estimator. " f"Must be one of {gufuncs.keys()}"
        )


def uv_weighted_score_weights(
    obs, fct, a=float("-inf"), b=float("inf"), w_func=None, backend=None
):
    """Calculate weights for a weighted scoring rule corresponding to the observations
    and ensemble members given a weight function."""
    B = backends.active if backend is None else backends[backend]
    if w_func is None:

        def w_func(x):
            return ((a <= x) & (x <= b)) * 1.0

    obs_w, fct_w = map(w_func, (obs, fct))
    obs_w, fct_w = map(B.asarray, (obs_w, fct_w))
    if B.any(obs_w < 0) or B.any(fct_w < 0):
        raise ValueError("`w_func` returns negative values")
    return obs_w, fct_w


def uv_weighted_score_chain(
    obs, fct, a=float("-inf"), b=float("inf"), v_func=None, backend=None
):
    """Calculate transformed observations and ensemble members for threshold-weighted scoring rules given a chaining function."""
    B = backends.active if backend is None else backends[backend]
    if v_func is None:
        a, b, obs, fct = map(B.asarray, (a, b, obs, fct))

        def v_func(x):
            return B.minimum(B.maximum(x, a), b)

    obs, fct = map(v_func, (obs, fct))
    return obs, fct


def mv_weighted_score_weights(obs, fct, w_func=None, backend=None):
    """Calculate weights for a weighted scoring rule corresponding to the observations
    and ensemble members given a weight function."""
    B = backends.active if backend is None else backends[backend]
    obs_w = B.apply_along_axis(w_func, obs, -1)
    fct_w = B.apply_along_axis(w_func, fct, -1)
    if backend == "torch":
        obs_w = obs_w * 1.0
        fct_w = fct_w * 1.0
    if B.any(obs_w < 0) or B.any(fct_w < 0):
        raise ValueError("`w_func` returns negative values")
    return obs_w, fct_w


def mv_weighted_score_chain(obs, fct, v_func=None):
    """Calculate transformed observations and ensemble members for threshold-weighted scoring rules given a chaining function."""
    obs, fct = map(v_func, (obs, fct))
    return obs, fct


def univariate_sort_ens(
    fct, ens_w=None, m_axis=-1, estimator=None, sorted_ensemble=False, backend=None
):
    """Sort ensemble members and weights if not already sorted."""
    B = backends.active if backend is None else backends[backend]
    sort_ensemble = not sorted_ensemble and estimator in ["qd", "pwm", "int"]
    if sort_ensemble:
        ind = B.argsort(fct, axis=-1)
        fct = B.gather(fct, ind, axis=-1)
    if ens_w is not None:
        ens_w = univariate_weight_check(ens_w, fct, m_axis, backend=backend)
        if sort_ensemble:
            ens_w = B.gather(ens_w, ind, axis=-1)
    return fct, ens_w


def nan_policy_check(nan_policy: str) -> None:
    """Validate the nan_policy argument."""
    valid = ("propagate", "omit", "raise")
    if nan_policy not in valid:
        raise ValueError(f"Invalid nan_policy '{nan_policy}'. Must be one of {valid}.")


def apply_nan_policy_ens_uv(
    obs: "Array",
    fct: "Array",
    nan_policy: "NanPolicy" = "propagate",
    ens_w: "Array | None" = None,
    estimator: str | None = None,
    m_axis: int = -1,
    backend: "Backend" = None,
) -> "tuple[Array, Array, Array | None]":
    """Apply a NaN policy to univariate ensemble forecasts (``fct`` shape ``..., M``).

    A member is treated as invalid if its forecast value **or** its weight
    (``ens_w``) is NaN. All policies return the same ``(obs, fct, ens_w)`` tuple
    so the caller can keep a fixed signature regardless of the policy. When
    supplied, ``ens_w`` is cast to float and returned with its ensemble axis
    (``m_axis``) moved to the last position, aligned with ``fct``.

    - ``'propagate'``: no-op; NaN members/weights flow through to a NaN result.
    - ``'raise'``: raise ``ValueError`` if any NaN is present in ``obs``,
      ``fct`` or ``ens_w``.
    - ``'omit'``: invalid members are given zero weight. NaN forecast values are
      replaced with ``0.0`` purely to keep them out of the arithmetic — their
      value is irrelevant once the corresponding weight is zero. ``ens_w`` is
      returned as a 0/1 mask when no weights were supplied, otherwise with the
      invalid entries zeroed.
    """
    B = backends.active if backend is None else backends[backend]

    if ens_w is not None:
        # align the ensemble axis with the (already permuted) forecasts — whose
        # ensemble axis is last — and cast to float so isnan works on integer
        # weights. Doing this here means the axis is moved exactly once.
        ens_w = B.moveaxis(B.asarray(ens_w, dtype=fct.dtype), m_axis, -1)

    if nan_policy == "propagate":
        return obs, fct, ens_w

    nan_mask = B.isnan(fct)
    if ens_w is not None:
        nan_mask = nan_mask | B.isnan(ens_w)

    if nan_policy == "raise":
        if B.any(nan_mask) or B.any(B.isnan(obs)):
            raise ValueError(
                "NaN values encountered in input. "
                "Use nan_policy='propagate' or nan_policy='omit' to handle NaN values."
            )
        return obs, fct, ens_w

    if nan_policy == "omit":
        if estimator in ["int", "akr", "akr_circperm"]:
            raise NotImplementedError(
                f"NaN handling with nan_policy='omit' is not implemented for estimator '{estimator}'."
            )
        # zero only the NaN forecast values; zero-weighted members never
        # contribute, so their (now 0.0) value does not affect the score.
        fct = B.where(B.isnan(fct), B.asarray(0.0), fct)
        if ens_w is None:
            ens_w = B.asarray(~nan_mask, dtype=fct.dtype)
        else:
            ens_w = B.where(nan_mask, B.asarray(0.0), ens_w)
        return obs, fct, ens_w

    raise ValueError(
        f"Invalid nan_policy '{nan_policy}'. Must be one of 'propagate', 'omit', 'raise'."
    )


def apply_nan_policy_ens_mv(obs, fct, nan_policy="propagate", backend=None):
    """Apply NaN policy to multivariate ensemble forecasts (fct shape: ..., M, D).

    A NaN in any variable of an ensemble member marks the entire member as invalid.

    For 'propagate': no-op, returns (obs, fct, None).
    For 'raise': raises ValueError if any NaN in fct or obs.
    For 'omit': returns (obs, fct_zeroed, nan_mask) where nan_mask has shape
    (..., M) — True for invalid members — and NaN members are replaced with 0.0.
    """
    B = backends.active if backend is None else backends[backend]

    if nan_policy == "raise":
        if B.any(B.isnan(fct)) or B.any(B.isnan(obs)):
            raise ValueError(
                "NaN values encountered in input. "
                "Use nan_policy='propagate' or nan_policy='omit' to handle NaN values."
            )
        return obs, fct, None

    if nan_policy == "omit":
        nan_mask = B.any(B.isnan(fct), axis=-1)  # shape (..., M)
        fct = B.where(nan_mask[..., None], B.asarray(0.0), fct)
        return obs, fct, nan_mask

    # propagate
    return obs, fct, None


def lazy_gufunc_wrapper_uv(func):
    """
    Wrapper for lazy/dynamic generalized universal functions so
    they return a value instead of modifying an output array in-place.
    This is the univariate version, which assumes that the shape of the
    return value is the same as the first input argument (the observations).
    """

    def wrapper(*args):
        out = np.empty_like(args[0])
        func(*args, out)
        if out.shape == ():
            return out.item()
        else:
            return out

    return wrapper


def lazy_gufunc_wrapper_mv(func):
    """
    Wrapper for lazy/dynamic generalized universal functions so
    they return a value instead of modifying an output array in-place.
    This is the multivariate version, which assumes that the shape of the
    return value is reduced to the shape of the first input argument
    (the observations) along the last axis.
    """

    def wrapper(*args):
        out = np.empty_like(args[0][..., 0])
        func(*args, out)
        if out.shape == ():
            return out.item()
        else:
            return out

    return wrapper
