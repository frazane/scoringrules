import numpy as np

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
            f"{estimator} is not a valid estimator. " f"Must be one of {gufuncs.keys()}"
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


def _gufunc_out(args, shape):
    """Output buffer for a lazy gufunc: the result dtype of ``obs`` and ``fct``
    (the first two arguments), at least float, so integer observations still
    give float scores. Auxiliary arguments (weights, exponents, constants built
    by the caller) do not take part, matching the dtype of the array path."""
    return np.empty(shape, dtype=np.result_type(args[0], args[1], np.float32))


def lazy_gufunc_wrapper_uv(func):
    """
    Wrapper for lazy/dynamic generalized universal functions so
    they return a value instead of modifying an output array in-place.
    This is the univariate version, which assumes that the shape of the
    return value is the same as the first input argument (the observations).
    """

    def wrapper(*args):
        out = _gufunc_out(args, np.shape(args[0]))
        func(*args, out)
        if out.shape == ():
            return out[()]  # a numpy scalar of out's dtype
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
        out = _gufunc_out(args, np.shape(args[0])[:-1])
        func(*args, out)
        if out.shape == ():
            return out[()]  # a numpy scalar of out's dtype
        else:
            return out

    return wrapper
