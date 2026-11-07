import typing as tp

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array


def es_ensemble(obs: "Array", fct: "Array", estimator: str = "nrg", *, xp) -> "Array":
    """
    Compute the energy score based on a finite ensemble.

    The ensemble and variables axes are on the second last and last dimensions respectively.
    """
    if estimator == "nrg":
        out = _es_ensemble_nrg(obs, fct, xp=xp)
    elif estimator == "fair":
        out = _es_ensemble_fair(obs, fct, xp=xp)
    elif estimator == "akr":
        out = _es_ensemble_akr(obs, fct, xp=xp)
    elif estimator == "akr_circperm":
        out = _es_ensemble_akr_circperm(obs, fct, xp=xp)
    else:
        raise ValueError(
            f"For the energy score, {estimator} must be one of 'nrg', 'fair', 'akr', and 'akr_circperm'."
        )

    return out


def _es_ensemble_nrg(obs: "Array", fct: "Array", *, xp) -> "Array":
    """Compute the Energy Score for a finite ensemble."""
    M: int = fct.shape[-2]

    err_norm = xp.norm(fct - xp.expand_dims(obs, axis=-2), -1)
    E_1 = xp.sum(err_norm, axis=-1) / M

    spread_norm = xp.norm(
        xp.expand_dims(fct, axis=-3) - xp.expand_dims(fct, axis=-2), -1
    )
    E_2 = xp.sum(spread_norm, axis=(-2, -1)) / (M**2)

    return E_1 - 0.5 * E_2


def _es_ensemble_fair(obs: "Array", fct: "Array", *, xp) -> "Array":
    """Compute the fair Energy Score for a finite ensemble."""
    M: int = fct.shape[-2]

    err_norm = xp.norm(fct - xp.expand_dims(obs, axis=-2), -1)
    E_1 = xp.sum(err_norm, axis=-1) / M

    spread_norm = xp.norm(
        xp.expand_dims(fct, axis=-3) - xp.expand_dims(fct, axis=-2), -1
    )
    E_2 = xp.sum(spread_norm, axis=(-2, -1)) / (M * (M - 1))

    return E_1 - 0.5 * E_2


def _es_ensemble_akr(obs: "Array", fct: "Array", *, xp) -> "Array":
    """Compute the Energy Score for a finite ensemble using the approximate kernel representation."""
    M: int = fct.shape[-2]

    err_norm = xp.norm(fct - xp.expand_dims(obs, axis=-2), -1)
    E_1 = xp.sum(err_norm, axis=-1) / M

    spread_norm = xp.norm(fct - xp.roll(fct, shift=1, axis=-2), -1)
    E_2 = xp.sum(spread_norm, axis=-1) / M

    return E_1 - 0.5 * E_2


def _es_ensemble_akr_circperm(obs: "Array", fct: "Array", *, xp) -> "Array":
    """Compute the Energy Score for a finite ensemble using the AKR with cyclic permutation."""
    M: int = fct.shape[-2]

    err_norm = xp.norm(fct - xp.expand_dims(obs, axis=-2), -1)
    E_1 = xp.sum(err_norm, axis=-1) / M

    # member i is paired with member (i + M // 2) % M
    spread_norm = xp.norm(fct - xp.roll(fct, shift=-(M // 2), axis=-2), -1)
    E_2 = xp.sum(spread_norm, axis=-1) / M

    return E_1 - 0.5 * E_2


def owes_ensemble(
    obs: "Array",  # (... D)
    fct: "Array",  # (... M D)
    ow: "Array",  # (...)
    fw: "Array",  # (... M)
    *,
    xp,
) -> "Array":
    """Compute the outcome-weighted energy score based on a finite ensemble."""
    wbar = xp.mean(fw, axis=-1)

    err_norm = xp.norm(fct - xp.expand_dims(obs, axis=-2), -1)  # (... M)
    E_1 = xp.mean(err_norm * fw * xp.expand_dims(ow, axis=-1), axis=-1) / wbar  # (...)

    spread_norm = xp.norm(
        xp.expand_dims(fct, axis=-2) - xp.expand_dims(fct, axis=-3), -1
    )  # (... M M)
    fw_prod = xp.expand_dims(fw, axis=-1) * xp.expand_dims(fw, axis=-2)  # (... M M)
    # out of place: torch autograd needs the norm output unchanged
    spread_norm = spread_norm * (fw_prod * ow[..., None, None])  # (... M M)
    E_2 = xp.mean(spread_norm, axis=(-2, -1)) / (wbar**2)  # (...)

    return E_1 - 0.5 * E_2


def vres_ensemble(
    obs: "Array",
    fct: "Array",
    ow: "Array",
    fw: "Array",
    *,
    xp,
) -> "Array":
    """Compute the vertically re-scaled energy score based on a finite ensemble."""
    wbar = xp.mean(fw, axis=-1)

    err_norm = xp.norm(fct - xp.expand_dims(obs, axis=-2), -1)  # (... M)
    err_norm = err_norm * (fw * xp.expand_dims(ow, axis=-1))  # (... M)
    E_1 = xp.mean(err_norm, axis=-1)  # (...)

    spread_norm = xp.norm(
        xp.expand_dims(fct, axis=-2) - xp.expand_dims(fct, axis=-3), -1
    )  # (... M M)
    fw_prod = xp.expand_dims(fw, axis=-2) * xp.expand_dims(fw, axis=-1)  # (... M M)
    E_2 = xp.mean(spread_norm * fw_prod, axis=(-2, -1))  # (...)

    rhobar = xp.mean(xp.norm(fct, -1) * fw, axis=-1)  # (...)
    E_3 = (rhobar - xp.norm(obs, -1) * ow) * (wbar - ow)  # (...)
    return E_1 - 0.5 * E_2 + E_3
