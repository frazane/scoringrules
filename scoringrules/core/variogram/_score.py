import typing as tp

from scoringrules.backend import backends

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, Backend


def variogram_score(
    obs: "Array",  # (... D)
    fct: "Array",  # (... M D)
    p: float = 1,
    backend: "Backend" = None,
) -> "Array":
    """Compute the Variogram Score for a multivariate finite ensemble."""
    B = backends.active if backend is None else backends[backend]
    M: int = fct.shape[-2]
    fct_diff = B.expand_dims(fct, -2) - B.expand_dims(fct, -1)  # (... M D D)
    vfct = B.sum(B.abs(fct_diff) ** p, axis=-3) / M  # (... D D)
    obs_diff = B.expand_dims(obs, -2) - B.expand_dims(obs, -1)  # (... D D)
    vobs = B.abs(obs_diff) ** p  # (... D D)
    return B.sum((vobs - vfct) ** 2, axis=(-2, -1))  # (...)


def owvariogram_score(
    obs: "Array",
    fct: "Array",
    ow: "Array",
    fw: "Array",
    p: float = 1,
    backend: "Backend" = None,
) -> "Array":
    """Compute the Outcome-Weighted Variogram Score for a multivariate finite ensemble."""
    B = backends.active if backend is None else backends[backend]
    M: int = fct.shape[-2]
    wbar = B.mean(fw, axis=-1)

    fct_diff = B.expand_dims(fct, -2) - B.expand_dims(fct, -1)  # (... M D D)
    fct_diff = B.abs(fct_diff) ** p  # (... M D D)

    obs_diff = B.expand_dims(obs, -2) - B.expand_dims(obs, -1)  # (... D D)
    obs_diff = B.abs(obs_diff) ** p  # (... D D)
    del obs, fct

    E_1 = (fct_diff - B.expand_dims(obs_diff, -3)) ** 2  # (... M D D)
    E_1 = B.sum(E_1, axis=(-2, -1))  # (... M)
    E_1 = B.sum(E_1 * fw * B.expand_dims(ow, -1), axis=-1) / (M * wbar)  # (...)

    fct_diff_spread = B.expand_dims(fct_diff, -3) - B.expand_dims(
        fct_diff, -4
    )  # (... M M D D)
    fw_prod = B.expand_dims(fw, -2) * B.expand_dims(fw, -1)  # (... M M)
    E_2 = B.sum(fct_diff_spread**2, axis=(-2, -1))  # (... M M)
    E_2 *= fw_prod * B.expand_dims(ow, (-2, -1))  # (... M M)
    E_2 = B.sum(E_2, axis=(-2, -1)) / (M**2 * wbar**2)  # (...)

    return E_1 - 0.5 * E_2


def vrvariogram_score(
    obs: "Array",
    fct: "Array",
    ow: "Array",
    fw: "Array",
    p: float = 1,
    backend: "Backend" = None,
) -> "Array":
    """Compute the Vertically Re-scaled Variogram Score for a multivariate finite ensemble."""
    B = backends.active if backend is None else backends[backend]
    M: int = fct.shape[-2]
    wbar = B.mean(fw, axis=-1)

    fct_diff = (
        B.abs(B.expand_dims(fct, -2) - B.expand_dims(fct, -1)) ** p
    )  # (... M D D)
    obs_diff = B.abs(B.expand_dims(obs, -2) - B.expand_dims(obs, -1)) ** p  # (... D D)

    E_1 = (fct_diff - B.expand_dims(obs_diff, axis=-3)) ** 2  # (... M D D)
    E_1 = B.sum(E_1, axis=(-2, -1))  # (... M)
    E_1 = B.sum(E_1 * fw * B.expand_dims(ow, axis=-1), axis=-1) / M  # (...)

    E_2 = (
        B.expand_dims(fct_diff, -3) - B.expand_dims(fct_diff, -4)
    ) ** 2  # (... M M D D)
    E_2 = B.sum(E_2, axis=(-2, -1))  # (... M M)

    fw_prod = B.expand_dims(fw, axis=-2) * B.expand_dims(fw, axis=-1)  # (... M M)
    E_2 = B.sum(E_2 * fw_prod, axis=(-2, -1)) / (M**2)  # (...)

    E_3 = B.sum(fct_diff**2, axis=(-2, -1))  # (... M)
    E_3 = B.sum(E_3 * fw, axis=-1) / M  # (...)
    E_3 -= B.sum(obs_diff**2, axis=(-2, -1)) * ow  # (...)
    E_3 *= wbar - ow  # (...)

    return E_1 - 0.5 * E_2 + E_3
