import typing as tp

from scoringrules.core.typing import Array, ArrayLike
from scoringrules.backend import backends


def energy_score(fcts: Array, obs: Array, m_axis=-2, v_axis=-1, backend=None) -> Array:
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
    E_2 = B.sum(spread_norm, axis=(m_axis, m_axis - 1)) / (M**2)
    del spread_norm, ens_diff
    return B.squeeze(E_1 - 0.5 * E_2)


def owenergy_score(
    fcts: Array,
    obs: Array,
    w_func: tp.Callable[[ArrayLike], ArrayLike],
    m_axis=-2,
    v_axis=-1,
    backend: str | None = None,
) -> Array:
    """Compute the outcome-weighted energy score based on a finite ensemble."""
    B = backends.active if backend is None else backends[backend]
    M = fcts.shape[m_axis]
    fw, ow = map(w_func, (fcts, obs))
    wbar = B.mean(fw, axis=-1, keepdims=True)

    err_norm = B.norm(
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


def vrenergy_score(
    fcts: Array,
    obs: Array,
    w_func: tp.Callable[[ArrayLike], ArrayLike],
    m_axis=-2,
    v_axis=-1,
    backend: str | None = None,
) -> Array:
    """Compute the vertically re-scaled energy score based on a finite ensemble."""
    B = backends.active if backend is None else backends[backend]
    M: int = fcts.shape[m_axis]

    fw, ow = map(w_func, (fcts, obs))
    wbar = B.mean(fw, axis=-1)

    err_norm = B.norm(
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
