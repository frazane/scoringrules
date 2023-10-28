import typing as tp

from scoringrules.core.typing import Array
from scoringrules.new_backend import backends, _NUMBA_IMPORTED


def energy_score(
    fcts: Array, obs: Array, m_axis=-2, v_axis=-1, backend= None
) -> Array:
    """
    Compute the energy score based on a finite ensemble.

    Note: there's a dirty trick to avoid computing the norm on a vector that contains
    zeros, because the gradient of the norm of zero returns nans.
    """
    B = backends.active if backend is None else backends[backend]
    M: int = fcts.shape[m_axis]
    err_norm = B.norm(
        fcts - B.expand_dims(obs, axis=m_axis), axis=v_axis, keepdims=True
    )
    E_1 = B.sum(err_norm, axis=m_axis) / M
    del err_norm
    ens_diff = B.expand_dims(fcts, axis=m_axis) - B.expand_dims(fcts, axis=(m_axis - 1))
    spread_norm = B.norm(ens_diff, axis=v_axis, keepdims=True)
    del ens_diff
    E_2 = B.sum(spread_norm, axis=(m_axis, m_axis - 1)) / (M**2)
    del spread_norm
    return B.squeeze(E_1 - 0.5 * E_2)


def owenergy_score(
    fcts: Array,
    obs: Array,
    w_func: tp.Callable = lambda x, *args: 1.0,
    w_funcargs: tuple = (),
    m_axis=-2,
    v_axis=-1,
    backend: str | None = None,
) -> Array:
    """Compute the outcome-weighted energy score based on a finite ensemble."""
    B = backends.active if backend is None else backends[backend]
    M: int = fcts.shape[m_axis]

    w_func_wrap = lambda x: w_func(x, *w_funcargs)
    fw = B.apply_along_axis(w_func_wrap, axis=-1, arr=fcts)
    ow = B.apply_along_axis(w_func_wrap, axis=-1, arr=obs)
    wbar = B.mean(fw, axis=-1, keepdims=True)

    err_norm = B.linalg.norm(
        fcts - B.expand_dims(obs, axis=m_axis), axis=v_axis, keepdims=True
    )
    E_1 = B.sum(
        err_norm
        * B.expand_dims(fw, axis=v_axis)
        * B.expand_dims(ow, axis=(m_axis, v_axis)),
        axis=m_axis,
    ) / (M * wbar)

    ens_diff = B.expand_dims(fcts, axis=m_axis) - B.expand_dims(fcts, axis=(m_axis - 1))
    spread_norm = B.norm(ens_diff, axis=v_axis, keepdims=True)

    fw_diff = B.expand_dims(fw, axis=m_axis) * B.expand_dims(fw, axis=v_axis)
    E_2 = B.sum(
        spread_norm * B.expand_dims(fw_diff, axis=-1) * ow[..., None, None, None],
        axis=(m_axis, m_axis - 1),
    ) / (M**2 * wbar**2)
    return B.squeeze(E_1 - 0.5 * E_2)


def twenergy_score(
    fcts: Array,
    obs: Array,
    v_func: tp.Callable = lambda x, *args: x,
    v_funcargs: tuple = (),
    m_axis=-2,
    v_axis=-1,
    backend: str | None = None,
) -> Array:
    """Compute the threshold-weighted energy score based on a finite ensemble."""
    B = backends.active if backend is None else backends[backend]
    if m_axis != -2:
        fcts = B.moveaxis(fcts, m_axis, -2)
        v_axis = v_axis - 1 if v_axis != 0 else v_axis
    if v_axis != -1:
        fcts = B.moveaxis(fcts, v_axis, -1)
        obs = B.moveaxis(obs, v_axis, -1)

    v_fcts = v_func(fcts, *v_funcargs)
    v_obs = v_func(obs, *v_funcargs)
    return self.energy_score(v_fcts, v_obs)


def vrenergy_score(
    fcts: Array,
    obs: Array,
    w_func: tp.Callable = lambda x, *args: 1.0,
    w_funcargs: tuple = (),
    m_axis=-2,
    v_axis=-1,
    backend: str | None = None,
) -> Array:
    """Compute the vertically re-scaled energy score based on a finite ensemble."""
    B = backends.active if backend is None else backends[backend]
    M: int = fcts.shape[m_axis]

    w_func_wrap = lambda x: w_func(x, *w_funcargs)
    fw = B.apply_along_axis(w_func_wrap, axis=-1, arr=fcts)
    ow = B.apply_along_axis(w_func_wrap, axis=-1, arr=obs)
    wbar = B.mean(fw, axis=-1)

    err_norm = B.linalg.norm(
        fcts - B.expand_dims(obs, axis=m_axis), axis=v_axis, keepdims=True
    )
    E_1 = (
        B.sum(
            err_norm
            * B.expand_dims(fw, axis=v_axis)
            * B.expand_dims(ow, axis=(m_axis, v_axis)),
            axis=m_axis,
        )
        / M
    )

    ens_diff = B.expand_dims(fcts, axis=m_axis) - B.expand_dims(fcts, axis=(m_axis - 1))
    spread_norm = B.norm(ens_diff, axis=v_axis, keepdims=True)

    fw_diff = B.expand_dims(fw, axis=m_axis) * B.expand_dims(fw, axis=v_axis)
    E_2 = B.sum(
        spread_norm * B.expand_dims(fw_diff, axis=-1),
        axis=(m_axis, m_axis - 1),
    ) / (M**2)

    rhobar = B.mean(B.norm(fcts, axis=v_axis) * fw, axis=(m_axis + 1))
    E_3 = (rhobar - B.norm(obs, axis=-1) * ow) * (wbar - ow)
    return B.squeeze(E_1 - 0.5 * E_2 + B.expand_dims(E_3, axis=-1))
