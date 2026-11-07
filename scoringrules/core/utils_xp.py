# scoringrules/core/utils_xp.py
"""xp-parameterised copies of the univariate ensemble helpers used by the
migrated CRPS family. Parallel to core/utils.py during the array-API transition;
the follow-on migration consolidates them."""

import typing as tp

from array_api_compat import is_array_api_obj

from .utils import EPSILON, _K_AXIS, _M_AXIS_UV, _shape_compatibility_check

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, NanPolicy  # noqa: F401


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
    inputs). Arrays are returned unchanged and ``None`` is passed through.
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


def univariate_weight_check(ens_w, fct, m_axis, xp):
    if ens_w is not None:
        ens_w = xp.asarray(ens_w)
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
    if sort_ensemble:
        ind = xp.argsort(fct, axis=-1)
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
