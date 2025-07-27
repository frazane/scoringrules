import typing as tp

from scoringrules.backend import backends

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, Backend


def es_ensemble(
    obs: "Array", fct: "Array", estimator: str = "nrg", backend=None
) -> "Array":
    """
    Compute the energy score based on a finite ensemble.

    The ensemble and variables axes are on the second last and last dimensions respectively.
    """
    if estimator == "nrg":
        out = _es_ensemble_nrg(obs, fct, backend=backend)
    elif estimator == "fair":
        out = _es_ensemble_fair(obs, fct, backend=backend)
    elif estimator == "akr":
        out = _es_ensemble_akr(obs, fct, backend=backend)
    elif estimator == "akr_circperm":
        out = _es_ensemble_akr_circperm(obs, fct, backend=backend)
    else:
        raise ValueError(
            f"{estimator} must be one of 'nrg' and 'fair' for the energy score."
        )

    return out


def _es_ensemble_nrg(obs: "Array", fct: "Array", backend: "Backend" = None) -> "Array":
    """Compute the Energy Score for a finite ensemble."""
    B = backends.active if backend is None else backends[backend]
    M: int = fct.shape[-2]

    err_norm = B.norm(fct - B.expand_dims(obs, -2), -1)
    E_1 = B.sum(err_norm, -1) / M

    spread_norm = B.norm(B.expand_dims(fct, -3) - B.expand_dims(fct, -2), -1)
    E_2 = B.sum(spread_norm, (-2, -1)) / (M**2)

    return E_1 - 0.5 * E_2


def _es_ensemble_fair(obs: "Array", fct: "Array", backend: "Backend" = None) -> "Array":
    """Compute the fair Energy Score for a finite ensemble."""
    B = backends.active if backend is None else backends[backend]
    M: int = fct.shape[-2]

    err_norm = B.norm(fct - B.expand_dims(obs, -2), -1)
    E_1 = B.sum(err_norm, -1) / M

    spread_norm = B.norm(B.expand_dims(fct, -3) - B.expand_dims(fct, -2), -1)
    E_2 = B.sum(spread_norm, (-2, -1)) / (M * (M - 1))

    return E_1 - 0.5 * E_2


def _es_ensemble_akr(obs: "Array", fct: "Array", backend: "Backend" = None) -> "Array":
    """Compute the Energy Score for a finite ensemble using the approximate kernel representation."""
    B = backends.active if backend is None else backends[backend]
    M: int = fct.shape[-2]

    err_norm = B.norm(fct - B.expand_dims(obs, -2), -1)
    E_1 = B.sum(err_norm, -1) / M

    spread_norm = B.norm(fct - B.roll(fct, shift=1, axis=-2), -1)
    E_2 = B.sum(spread_norm, -1) / M

    return E_1 - 0.5 * E_2


def _es_ensemble_akr_circperm(
    obs: "Array", fct: "Array", backend: "Backend" = None
) -> "Array":
    """Compute the Energy Score for a finite ensemble using the AKR with cyclic permutation."""
    B = backends.active if backend is None else backends[backend]
    M: int = fct.shape[-2]

    err_norm = B.norm(fct - B.expand_dims(obs, -2), -1)
    E_1 = B.sum(err_norm, -1) / M

    shift = M // 2
    spread_norm = B.norm(fct - B.roll(fct, shift=shift, axis=-2), -1)
    E_2 = B.sum(spread_norm, -1) / M

    return E_1 - 0.5 * E_2


def owes_ensemble(
    obs: "Array",  # (... D)
    fct: "Array",  # (... M D)
    ow: "Array",  # (...)
    fw: "Array",  # (... M)
    backend: "Backend" = None,
) -> "Array":
    """Compute the outcome-weighted energy score based on a finite ensemble."""
    B = backends.active if backend is None else backends[backend]
    M = fct.shape[-2]
    wbar = B.sum(fw, -1) / M

    err_norm = B.norm(fct - B.expand_dims(obs, -2), -1)  # (... M)
    E_1 = B.sum(err_norm * fw * B.expand_dims(ow, -1), -1) / (M * wbar)  # (...)

    spread_norm = B.norm(
        B.expand_dims(fct, -2) - B.expand_dims(fct, -3), -1
    )  # (... M M)
    fw_prod = B.expand_dims(fw, -1) * B.expand_dims(fw, -2)  # (... M M)
    spread_norm *= fw_prod * B.expand_dims(ow, (-2, -1))  # (... M M)
    E_2 = B.sum(spread_norm, (-2, -1)) / (M**2 * wbar**2)  # (...)

    return E_1 - 0.5 * E_2


def vres_ensemble(
    obs: "Array",
    fct: "Array",
    ow: "Array",
    fw: "Array",
    backend: "Backend" = None,
) -> "Array":
    """Compute the vertically re-scaled energy score based on a finite ensemble."""
    B = backends.active if backend is None else backends[backend]
    M, D = fct.shape[-2:]
    wbar = B.sum(fw, -1) / M

    err_norm = B.norm(fct - B.expand_dims(obs, -2), -1)  # (... M)
    err_norm *= fw * B.expand_dims(ow, -1)  # (... M)
    E_1 = B.sum(err_norm, -1) / M  # (...)

    spread_norm = B.norm(
        B.expand_dims(fct, -2) - B.expand_dims(fct, -3), -1
    )  # (... M M)
    fw_prod = B.expand_dims(fw, -2) * B.expand_dims(fw, -1)  # (... M M)
    E_2 = B.sum(spread_norm * fw_prod, (-2, -1)) / (M**2)  # (...)

    rhobar = B.sum(B.norm(fct, -1) * fw, -1) / M  # (...)
    E_3 = (rhobar - B.norm(obs, -1) * ow) * (wbar - ow)  # (...)
    return E_1 - 0.5 * E_2 + E_3
