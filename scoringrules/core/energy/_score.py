import typing as tp

from scoringrules.backend import backends

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array


def energy_score(
    fcts: "Array", obs: "Array", backend=None  # (... M D)  # (... D)
) -> "Array":
    """
    Compute the energy score based on a finite ensemble.

    The ensemble and variables axes are on the second last and last dimensions respectively.
    """
    B = backends.active if backend is None else backends[backend]
    M: int = fcts.shape[-2]

    err_norm = B.norm(fcts - B.expand_dims(obs, -2), -1)  # (... M)
    E_1 = B.sum(err_norm, -1) / M  # (...)

    spread_norm = B.norm(
        B.expand_dims(fcts, -3) - B.expand_dims(fcts, -2), -1
    )  # (... M M)
    E_2 = B.sum(spread_norm, (-2, -1)) / (M**2)  # (...)
    return E_1 - 0.5 * E_2


def owenergy_score(
    fcts: "Array",  # (... M D)
    obs: "Array",  # (... D)
    fw: "Array",  # (... M)
    ow: "Array",  # (...)
    backend: str | None = None,
) -> "Array":
    """Compute the outcome-weighted energy score based on a finite ensemble."""
    B = backends.active if backend is None else backends[backend]
    M = fcts.shape[-2]
    wbar = B.sum(fw, -1) / M

    err_norm = B.norm(fcts - B.expand_dims(obs, -2), -1)  # (... M)
    E_1 = B.sum(err_norm * fw * B.expand_dims(ow, -1), -1) / (M * wbar)  # (...)

    spread_norm = B.norm(
        B.expand_dims(fcts, -2) - B.expand_dims(fcts, -3), -1
    )  # (... M M)
    fw_prod = B.expand_dims(fw, -1) * B.expand_dims(fw, -2)  # (... M M)
    spread_norm *= fw_prod * B.expand_dims(ow, (-2, -1))  # (... M M)
    E_2 = B.sum(spread_norm, (-2, -1)) / (M**2 * wbar**2)  # (...)

    return E_1 - 0.5 * E_2


def vrenergy_score(
    fcts: "Array",
    obs: "Array",
    fw: "Array",
    ow: "Array",
    backend: str | None = None,
) -> "Array":
    """Compute the vertically re-scaled energy score based on a finite ensemble."""
    B = backends.active if backend is None else backends[backend]
    M, D = fcts.shape[-2:]
    wbar = B.sum(fw, -1) / M

    err_norm = B.norm(fcts - B.expand_dims(obs, -2), -1)  # (... M)
    err_norm *= fw * B.expand_dims(ow, -1)  # (... M)
    E_1 = B.sum(err_norm, -1) / M  # (...)

    spread_norm = B.norm(
        B.expand_dims(fcts, -2) - B.expand_dims(fcts, -3), -1
    )  # (... M M)
    fw_prod = B.expand_dims(fw, -2) * B.expand_dims(fw, -1)  # (... M M)
    E_2 = B.sum(spread_norm * fw_prod, (-2, -1)) / (M**2)  # (...)

    rhobar = B.sum(B.norm(fcts, -1) * fw, -1) / M  # (...)
    E_3 = (rhobar - B.norm(obs, -1) * ow) * (wbar - ow)  # (...)
    return E_1 - 0.5 * E_2 + E_3
