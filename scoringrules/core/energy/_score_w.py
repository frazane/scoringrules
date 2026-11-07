import typing as tp

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array


def es_ensemble_w(
    obs: "Array", fct: "Array", ens_w: "Array", estimator: str = "nrg", *, xp
) -> "Array":
    """
    Compute the energy score based on a finite ensemble.

    The ensemble and variables axes are on the second last and last dimensions respectively.
    """
    if estimator == "nrg":
        out = _es_ensemble_nrg_w(obs, fct, ens_w, xp=xp)
    elif estimator == "fair":
        out = _es_ensemble_fair_w(obs, fct, ens_w, xp=xp)
    elif estimator == "akr":
        out = _es_ensemble_akr_w(obs, fct, ens_w, xp=xp)
    elif estimator == "akr_circperm":
        out = _es_ensemble_akr_circperm_w(obs, fct, ens_w, xp=xp)
    else:
        raise ValueError(
            f"For the energy score, {estimator} must be one of 'nrg', 'fair', 'akr', and 'akr_circperm'."
        )

    return out


def _es_ensemble_nrg_w(obs: "Array", fct: "Array", ens_w: "Array", *, xp) -> "Array":
    """
    Compute the energy score based on a finite ensemble with the energy estimator.
    """
    err_norm = xp.norm(fct - xp.expand_dims(obs, axis=-2), -1)
    E_1 = xp.sum(err_norm * ens_w, axis=-1)

    spread_norm = xp.norm(
        xp.expand_dims(fct, axis=-3) - xp.expand_dims(fct, axis=-2), -1
    )
    E_2 = xp.sum(
        spread_norm * xp.expand_dims(ens_w, axis=-1) * xp.expand_dims(ens_w, axis=-2),
        axis=(-2, -1),
    )

    return E_1 - 0.5 * E_2


def _es_ensemble_fair_w(obs: "Array", fct: "Array", ens_w: "Array", *, xp) -> "Array":
    """
    Compute the energy score based on a finite ensemble with the fair estimator.
    """
    err_norm = xp.norm(fct - xp.expand_dims(obs, axis=-2), -1)
    E_1 = xp.sum(err_norm * ens_w, axis=-1)

    spread_norm = xp.norm(
        xp.expand_dims(fct, axis=-3) - xp.expand_dims(fct, axis=-2), -1
    )
    E_2 = xp.sum(
        spread_norm * xp.expand_dims(ens_w, axis=-1) * xp.expand_dims(ens_w, axis=-2),
        axis=(-2, -1),
    )

    fair_c = 1 - xp.sum(ens_w * ens_w, axis=-1)

    return E_1 - 0.5 * E_2 / fair_c


def _es_ensemble_akr_w(obs: "Array", fct: "Array", ens_w: "Array", *, xp) -> "Array":
    """Compute the Energy Score for a finite ensemble using the approximate kernel representation."""
    err_norm = xp.norm(fct - xp.expand_dims(obs, axis=-2), -1)
    E_1 = xp.sum(err_norm * ens_w, axis=-1)

    spread_norm = xp.norm(fct - xp.roll(fct, shift=1, axis=-2), -1)
    E_2 = xp.sum(spread_norm * ens_w, axis=-1)

    return E_1 - 0.5 * E_2


def _es_ensemble_akr_circperm_w(
    obs: "Array", fct: "Array", ens_w: "Array", *, xp
) -> "Array":
    """Compute the Energy Score for a finite ensemble using the AKR with cyclic permutation."""
    M: int = fct.shape[-2]

    err_norm = xp.norm(fct - xp.expand_dims(obs, axis=-2), -1)
    E_1 = xp.sum(err_norm * ens_w, axis=-1)

    # member i is paired with member (i + M // 2) % M
    spread_norm = xp.norm(fct - xp.roll(fct, shift=-(M // 2), axis=-2), -1)
    E_2 = xp.sum(spread_norm * ens_w, axis=-1)

    return E_1 - 0.5 * E_2


def owes_ensemble_w(
    obs: "Array",  # (... D)
    fct: "Array",  # (... M D)
    ow: "Array",  # (...)
    fw: "Array",  # (... M)
    ens_w: "Array",  # (... M)
    *,
    xp,
) -> "Array":
    """Compute the outcome-weighted energy score based on a finite ensemble."""
    wbar = xp.sum(fw * ens_w, axis=-1)

    err_norm = xp.norm(fct - xp.expand_dims(obs, axis=-2), -1)  # (... M)
    E_1 = (
        xp.sum(err_norm * fw * xp.expand_dims(ow, axis=-1) * ens_w, axis=-1) / wbar
    )  # (...)

    spread_norm = xp.norm(
        xp.expand_dims(fct, axis=-2) - xp.expand_dims(fct, axis=-3), -1
    )  # (... M M)
    fw_prod = xp.expand_dims(fw, axis=-1) * xp.expand_dims(fw, axis=-2)  # (... M M)
    # out of place: torch autograd needs the norm output unchanged
    spread_norm = spread_norm * (fw_prod * ow[..., None, None])  # (... M M)
    E_2 = xp.sum(
        spread_norm * xp.expand_dims(ens_w, axis=-1) * xp.expand_dims(ens_w, axis=-2),
        axis=(-2, -1),
    ) / (wbar**2)  # (...)

    return E_1 - 0.5 * E_2


def vres_ensemble_w(
    obs: "Array",
    fct: "Array",
    ow: "Array",
    fw: "Array",
    ens_w: "Array",
    *,
    xp,
) -> "Array":
    """Compute the vertically re-scaled energy score based on a finite ensemble."""
    wbar = xp.sum(fw * ens_w, axis=-1)

    err_norm = xp.norm(fct - xp.expand_dims(obs, axis=-2), -1)  # (... M)
    err_norm = err_norm * (fw * xp.expand_dims(ow, axis=-1))  # (... M)
    E_1 = xp.sum(err_norm * ens_w, axis=-1)  # (...)

    spread_norm = xp.norm(
        xp.expand_dims(fct, axis=-2) - xp.expand_dims(fct, axis=-3), -1
    )  # (... M M)
    fw_prod = xp.expand_dims(fw, axis=-2) * xp.expand_dims(fw, axis=-1)  # (... M M)
    E_2 = xp.sum(
        spread_norm
        * fw_prod
        * xp.expand_dims(ens_w, axis=-1)
        * xp.expand_dims(ens_w, axis=-2),
        axis=(-2, -1),
    )  # (...)

    rhobar = xp.sum(xp.norm(fct, -1) * fw * ens_w, axis=-1)  # (...)
    E_3 = (rhobar - xp.norm(obs, -1) * ow) * (wbar - ow)  # (...)
    return E_1 - 0.5 * E_2 + E_3
