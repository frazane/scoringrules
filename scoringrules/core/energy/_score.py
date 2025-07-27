import typing as tp

from scoringrules.backend import backends

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, Backend


def es_ensemble(obs: "Array", fct: "Array", ens_w: "Array", backend=None) -> "Array":
    """
    Compute the energy score based on a finite ensemble.

    The ensemble and variables axes are on the second last and last dimensions respectively.
    """
    B = backends.active if backend is None else backends[backend]

    err_norm = B.norm(fct - B.expand_dims(obs, -2), -1)
    E_1 = B.sum(err_norm * ens_w, -1)

    spread_norm = B.norm(B.expand_dims(fct, -3) - B.expand_dims(fct, -2), -1)
    E_2 = B.sum(
        spread_norm * B.expand_dims(ens_w, -1) * B.expand_dims(ens_w, -2), (-2, -1)
    )

    return E_1 - 0.5 * E_2


def owes_ensemble(
    obs: "Array",  # (... D)
    fct: "Array",  # (... M D)
    ow: "Array",  # (...)
    fw: "Array",  # (... M)
    ens_w: "Array",  # (... M)
    backend: "Backend" = None,
) -> "Array":
    """Compute the outcome-weighted energy score based on a finite ensemble."""
    B = backends.active if backend is None else backends[backend]
    wbar = B.sum(fw * ens_w, -1)

    err_norm = B.norm(fct - B.expand_dims(obs, -2), -1)  # (... M)
    E_1 = B.sum(err_norm * fw * B.expand_dims(ow, -1) * ens_w, -1) / wbar  # (...)

    spread_norm = B.norm(
        B.expand_dims(fct, -2) - B.expand_dims(fct, -3), -1
    )  # (... M M)
    fw_prod = B.expand_dims(fw, -1) * B.expand_dims(fw, -2)  # (... M M)
    spread_norm *= fw_prod * B.expand_dims(ow, (-2, -1))  # (... M M)
    E_2 = B.sum(
        spread_norm * B.expand_dims(ens_w, -1) * B.expand_dims(ens_w, -2), (-2, -1)
    ) / (wbar**2)  # (...)

    return E_1 - 0.5 * E_2


def vres_ensemble(
    obs: "Array",
    fct: "Array",
    ow: "Array",
    fw: "Array",
    ens_w: "Array",
    backend: "Backend" = None,
) -> "Array":
    """Compute the vertically re-scaled energy score based on a finite ensemble."""
    B = backends.active if backend is None else backends[backend]
    wbar = B.sum(fw * ens_w, -1)

    err_norm = B.norm(fct - B.expand_dims(obs, -2), -1)  # (... M)
    err_norm *= fw * B.expand_dims(ow, -1)  # (... M)
    E_1 = B.sum(err_norm * ens_w, -1)  # (...)

    spread_norm = B.norm(
        B.expand_dims(fct, -2) - B.expand_dims(fct, -3), -1
    )  # (... M M)
    fw_prod = B.expand_dims(fw, -2) * B.expand_dims(fw, -1)  # (... M M)
    E_2 = B.sum(
        spread_norm * fw_prod * B.expand_dims(ens_w, -1) * B.expand_dims(ens_w, -2),
        (-2, -1),
    )  # (...)

    rhobar = B.sum(B.norm(fct, -1) * fw * ens_w, -1)  # (...)
    E_3 = (rhobar - B.norm(obs, -1) * ow) * (wbar - ow)  # (...)
    return E_1 - 0.5 * E_2 + E_3
