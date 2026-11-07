import typing as tp

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array


def vs_ensemble(
    obs: "Array",  # (... D)
    fct: "Array",  # (... M D)
    w: "Array",  # (..., D D)
    p: float = 0.5,
    estimator: str = "nrg",
    *,
    xp,
) -> "Array":
    """Compute the Variogram Score for a multivariate finite ensemble."""
    M: int = fct.shape[-2]

    fct_diff = (
        xp.abs(xp.expand_dims(fct, axis=-2) - xp.expand_dims(fct, axis=-1)) ** p
    )  # (... M D D)
    obs_diff = (
        xp.abs(xp.expand_dims(obs, axis=-2) - xp.expand_dims(obs, axis=-1)) ** p
    )  # (... D D)

    if estimator == "nrg":
        fct_diff = xp.sum(fct_diff, axis=-3) / M  # (... D D)
        out = xp.sum(w * (obs_diff - fct_diff) ** 2, axis=(-2, -1))  # (...)

    elif estimator == "fair":
        E_1 = (fct_diff - xp.expand_dims(obs_diff, axis=-3)) ** 2  # (... M D D)
        E_1 = xp.sum(xp.expand_dims(w, axis=-3) * E_1, axis=(-2, -1))  # (... M)
        E_1 = xp.sum(E_1, axis=-1) / M  # (...)

        E_2 = (
            xp.expand_dims(fct_diff, axis=-3) - xp.expand_dims(fct_diff, axis=-4)
        ) ** 2  # (... M M D D)
        E_2 = xp.sum(w[..., None, None, :, :] * E_2, axis=(-2, -1))  # (... M M)
        E_2 = xp.sum(E_2, axis=(-2, -1)) / (M * (M - 1))  # (...)

        out = E_1 - 0.5 * E_2
    else:
        raise ValueError(
            f"For the variogram score, {estimator} must be one of 'nrg', and 'fair'."
        )

    return out


def owvs_ensemble(
    obs: "Array",
    fct: "Array",
    w: "Array",
    ow: "Array",
    fw: "Array",
    p: float = 0.5,
    *,
    xp,
) -> "Array":
    """Compute the Outcome-Weighted Variogram Score for a multivariate finite ensemble."""
    wbar = xp.mean(fw, axis=-1)

    fct_diff = (
        xp.abs(xp.expand_dims(fct, axis=-2) - xp.expand_dims(fct, axis=-1)) ** p
    )  # (... M D D)
    obs_diff = (
        xp.abs(xp.expand_dims(obs, axis=-2) - xp.expand_dims(obs, axis=-1)) ** p
    )  # (... D D)

    E_1 = (fct_diff - xp.expand_dims(obs_diff, axis=-3)) ** 2  # (... M D D)
    E_1 = xp.sum(xp.expand_dims(w, axis=-3) * E_1, axis=(-2, -1))  # (... M)
    E_1 = xp.mean(E_1 * fw * xp.expand_dims(ow, axis=-1), axis=-1) / wbar  # (...)

    fct_diff_spread = xp.expand_dims(fct_diff, axis=-3) - xp.expand_dims(
        fct_diff, -4
    )  # (... M M D D)
    fw_prod = xp.expand_dims(fw, axis=-2) * xp.expand_dims(fw, axis=-1)  # (... M M)
    E_2 = xp.sum(
        w[..., None, None, :, :] * fct_diff_spread**2, axis=(-2, -1)
    )  # (... M M)
    E_2 = E_2 * (fw_prod * ow[..., None, None])  # (... M M)
    E_2 = xp.mean(E_2, axis=(-2, -1)) / (wbar**2)  # (...)

    return E_1 - 0.5 * E_2


def vrvs_ensemble(
    obs: "Array",
    fct: "Array",
    w: "Array",
    ow: "Array",
    fw: "Array",
    p: float = 0.5,
    *,
    xp,
) -> "Array":
    """Compute the Vertically Re-scaled Variogram Score for a multivariate finite ensemble."""
    wbar = xp.mean(fw, axis=-1)

    fct_diff = (
        xp.abs(xp.expand_dims(fct, axis=-2) - xp.expand_dims(fct, axis=-1)) ** p
    )  # (... M D D)
    obs_diff = (
        xp.abs(xp.expand_dims(obs, axis=-2) - xp.expand_dims(obs, axis=-1)) ** p
    )  # (... D D)

    E_1 = (fct_diff - xp.expand_dims(obs_diff, axis=-3)) ** 2  # (... M D D)
    E_1 = xp.sum(xp.expand_dims(w, axis=-3) * E_1, axis=(-2, -1))  # (... M)
    E_1 = xp.mean(E_1 * fw * xp.expand_dims(ow, axis=-1), axis=-1)  # (...)

    E_2 = (
        xp.expand_dims(fct_diff, axis=-3) - xp.expand_dims(fct_diff, axis=-4)
    ) ** 2  # (... M M D D)
    E_2 = xp.sum(w[..., None, None, :, :] * E_2, axis=(-2, -1))  # (... M M)

    fw_prod = xp.expand_dims(fw, axis=-2) * xp.expand_dims(fw, axis=-1)  # (... M M)
    E_2 = xp.mean(E_2 * fw_prod, axis=(-2, -1))  # (...)

    E_3 = xp.sum(xp.expand_dims(w, axis=-3) * fct_diff**2, axis=(-2, -1))  # (... M)
    E_3 = xp.mean(E_3 * fw, axis=-1)  # (...)
    E_3 = E_3 - xp.sum(w * obs_diff**2, axis=(-2, -1)) * ow  # (...)
    E_3 = E_3 * (wbar - ow)  # (...)

    return E_1 - 0.5 * E_2 + E_3
