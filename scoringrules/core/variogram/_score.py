import typing as tp

from scoringrules.backend import backends

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array


def variogram_score(
    fcst: "Array",  # (... M D)
    obs: "Array",  # (... D)
    p: float = 1,
    backend: tp.Literal["numba", "numpy", "jax", "torch"] | None = None,
) -> "Array":
    """Compute the Variogram Score for a multivariate finite ensemble."""
    B = backends.active if backend is None else backends[backend]
    M: int = fcst.shape[-2]
    fcst_diff = B.expand_dims(fcst, -2) - B.expand_dims(fcst, -1)  # (... M D D)
    vfcts = B.sum(B.abs(fcst_diff) ** p, axis=-3) / M  # (... D D)
    obs_diff = B.expand_dims(obs, -2) - B.expand_dims(obs, -1)  # (... D D)
    vobs = B.abs(obs_diff) ** p  # (... D D)
    return B.sum((vobs - vfcts) ** 2, axis=(-2, -1))  # (...)


def owvariogram_score(
    fcst: "Array",
    obs: "Array",
    fw: "Array",
    ow: "Array",
    p: float = 1,
    backend: tp.Literal["numba", "numpy", "jax", "torch"] | None = None,
) -> "Array":
    """Compute the Outcome-Weighted Variogram Score for a multivariate finite ensemble."""
    B = backends.active if backend is None else backends[backend]
    M: int = fcst.shape[-2]
    wbar = B.mean(fw, axis=-1)

    fcst_diff = B.expand_dims(fcst, -2) - B.expand_dims(fcst, -1)  # (... M D D)
    fcst_diff = B.abs(fcst_diff) ** p  # (... M D D)

    obs_diff = B.expand_dims(obs, -2) - B.expand_dims(obs, -1)  # (... D D)
    obs_diff = B.abs(obs_diff) ** p  # (... D D)
    del fcst, obs

    E_1 = (fcst_diff - B.expand_dims(obs_diff, -3)) ** 2  # (... M D D)
    E_1 = B.sum(E_1, axis=(-2, -1))  # (... M)
    E_1 = B.sum(E_1 * fw * B.expand_dims(ow, -1), axis=-1) / (M * wbar)  # (...)

    fcst_diff_spread = B.expand_dims(fcst_diff, -3) - B.expand_dims(
        fcst_diff, -4
    )  # (... M M D D)
    fw_prod = B.expand_dims(fw, -2) * B.expand_dims(fw, -1)  # (... M M)
    E_2 = B.sum(fcst_diff_spread**2, axis=(-2, -1))  # (... M M)
    E_2 *= fw_prod * B.expand_dims(ow, (-2, -1))  # (... M M)
    E_2 = B.sum(E_2, axis=(-2, -1)) / (M**2 * wbar**2)  # (...)

    return E_1 - 0.5 * E_2


def vrvariogram_score(
    fcst: "Array",
    obs: "Array",
    fw: "Array",
    ow: "Array",
    p: float = 1,
    backend: tp.Literal["numba", "numpy", "jax", "torch"] | None = None,
) -> "Array":
    """Compute the Vertically Re-scaled Variogram Score for a multivariate finite ensemble."""
    B = backends.active if backend is None else backends[backend]
    M: int = fcst.shape[-2]
    wbar = B.mean(fw, axis=-1)

    fcst_diff = (
        B.abs(B.expand_dims(fcst, -2) - B.expand_dims(fcst, -1)) ** p
    )  # (... M D D)
    obs_diff = B.abs(B.expand_dims(obs, -2) - B.expand_dims(obs, -1)) ** p  # (... D D)

    E_1 = (fcst_diff - B.expand_dims(obs_diff, axis=-3)) ** 2  # (... M D D)
    E_1 = B.sum(E_1, axis=(-2, -1))  # (... M)
    E_1 = B.sum(E_1 * fw * B.expand_dims(ow, axis=-1), axis=-1) / M  # (...)

    E_2 = (
        B.expand_dims(fcst_diff, -3) - B.expand_dims(fcst_diff, -4)
    ) ** 2  # (... M M D D)
    E_2 = B.sum(E_2, axis=(-2, -1))  # (... M M)

    fw_prod = B.expand_dims(fw, axis=-2) * B.expand_dims(fw, axis=-1)  # (... M M)
    E_2 = B.sum(E_2 * fw_prod, axis=(-2, -1)) / (M**2)  # (...)

    E_3 = B.sum(fcst_diff**2, axis=(-2, -1))  # (... M)
    E_3 = B.sum(E_3 * fw, axis=-1) / M  # (...)
    E_3 -= B.sum(obs_diff**2, axis=(-2, -1)) * ow  # (...)
    E_3 *= wbar - ow  # (...)

    return E_1 - 0.5 * E_2 + E_3
