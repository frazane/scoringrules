from scoringrules.backend import backends

_V_AXIS = -1
_M_AXIS = -2


def _multivariate_shape_compatibility(fcts, obs, m_axis) -> None:
    f_shape = fcts.shape
    o_shape = obs.shape
    o_shape_broadcast = o_shape[:m_axis] + (f_shape[m_axis],) + o_shape[m_axis:]
    if o_shape_broadcast != f_shape:
        raise ValueError(
            f"Forecasts shape {f_shape} and observations shape {o_shape} are not compatible for broadcasting!"
        )


def _multivariate_shape_permute(fcts, obs, m_axis, v_axis, backend=None):
    B = backends.active if backend is None else backends[backend]
    v_axis_obs = v_axis - 1 if m_axis < v_axis else v_axis
    fcts = B.moveaxis(fcts, (m_axis, v_axis), (_M_AXIS, _V_AXIS))
    obs = B.moveaxis(obs, v_axis_obs, _V_AXIS)
    return fcts, obs


def multivariate_array_check(fcts, obs, m_axis, v_axis, backend=None):
    """Check and adapt the shapes of multivariate forecasts and observations arrays."""
    B = backends.active if backend is None else backends[backend]
    fcts, obs = map(B.asarray, (fcts, obs))
    m_axis = m_axis if m_axis >= 0 else fcts.ndim + m_axis
    v_axis = v_axis if v_axis >= 0 else fcts.ndim + v_axis
    _multivariate_shape_compatibility(fcts, obs, m_axis)
    return _multivariate_shape_permute(fcts, obs, m_axis, v_axis, backend=backend)
