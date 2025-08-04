import numpy as np

from scoringrules.backend import backends

_V_AXIS = -1
_M_AXIS = -2


def _multivariate_shape_compatibility(obs, fct, m_axis) -> None:
    f_shape = fct.shape
    o_shape = obs.shape
    o_shape_broadcast = o_shape[:m_axis] + (f_shape[m_axis],) + o_shape[m_axis:]
    if o_shape_broadcast != f_shape:
        raise ValueError(
            f"Forecasts shape {f_shape} and observations shape {o_shape} are not compatible for broadcasting!"
        )


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
    _multivariate_shape_compatibility(obs, fct, m_axis)
    return _multivariate_shape_permute(obs, fct, m_axis, v_axis, backend=backend)


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
        return out

    return wrapper
