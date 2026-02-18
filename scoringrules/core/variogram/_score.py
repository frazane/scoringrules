import typing as tp

from scoringrules.backend import backends

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, Backend


def vs_ensemble(
    obs: "Array",  # (... D)
    fct: "Array",  # (... M D)
    w: "Array",  # (..., D D)
    p: float = 1,
    estimator: str = "nrg",
    backend: "Backend" = None,
) -> "Array":
    """Compute the Variogram Score for a multivariate finite ensemble."""
    B = backends.active if backend is None else backends[backend]
    M: int = fct.shape[-2]

    fct_diff = (
        B.abs(B.expand_dims(fct, -2) - B.expand_dims(fct, -1)) ** p
    )  # (... M D D)
    obs_diff = B.abs(B.expand_dims(obs, -2) - B.expand_dims(obs, -1)) ** p  # (... D D)

    if estimator == "nrg":
        fct_diff = B.expand_dims(fct, -2) - B.expand_dims(fct, -1)  # (... M D D)
        vfct = B.mean(B.abs(fct_diff) ** p, axis=-3)  # (... D D)
        obs_diff = B.expand_dims(obs, -2) - B.expand_dims(obs, -1)  # (... D D)
        vobs = B.abs(obs_diff) ** p  # (... D D)
        out = B.sum(w * (vobs - vfct) ** 2, axis=(-2, -1))  # (...)

    elif estimator == "fair":
        E_1 = (fct_diff - B.expand_dims(obs_diff, axis=-3)) ** 2  # (... M D D)
        E_1 = B.sum(E_1, axis=(-2, -1))  # (... M)
        E_1 = B.sum(E_1, axis=-1) / M  # (...)

        E_2 = (
            B.expand_dims(fct_diff, -3) - B.expand_dims(fct_diff, -4)
        ) ** 2  # (... M M D D)
        E_2 = B.sum(E_2, axis=(-2, -1))  # (... M M)
        E_2 = B.sum(E_2, axis=(-2, -1)) / (M * (M - 1))  # (...)

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
    backend: "Backend" = None,
) -> "Array":
    """Compute the Outcome-Weighted Variogram Score for a multivariate finite ensemble."""
    B = backends.active if backend is None else backends[backend]
    wbar = B.mean(fw, axis=-1)

    fct_diff = (
        B.abs(B.expand_dims(fct, -2) - B.expand_dims(fct, -1)) ** p
    )  # (... M D D)
    obs_diff = B.abs(B.expand_dims(obs, -2) - B.expand_dims(obs, -1)) ** p  # (... D D)

    E_1 = (fct_diff - B.expand_dims(obs_diff, -3)) ** 2  # (... M D D)
    E_1 = B.sum(B.expand_dims(w, -3) * E_1, axis=(-2, -1))  # (... M)
    E_1 = B.mean(E_1 * fw * B.expand_dims(ow, -1), axis=-1) / wbar  # (...)

    fct_diff_spread = B.expand_dims(fct_diff, -3) - B.expand_dims(
        fct_diff, -4
    )  # (... M M D D)
    fw_prod = B.expand_dims(fw, -2) * B.expand_dims(fw, -1)  # (... M M)
    E_2 = B.sum(
        B.expand_dims(w, (-3, -4)) * fct_diff_spread**2, axis=(-2, -1)
    )  # (... M M)
    E_2 *= fw_prod * B.expand_dims(ow, (-2, -1))  # (... M M)
    E_2 = B.mean(E_2, axis=(-2, -1)) / (wbar**2)  # (...)

    return E_1 - 0.5 * E_2


def vrvs_ensemble(
    obs: "Array",
    fct: "Array",
    w: "Array",
    ow: "Array",
    fw: "Array",
    p: float = 0.5,
    backend: "Backend" = None,
) -> "Array":
    """Compute the Vertically Re-scaled Variogram Score for a multivariate finite ensemble."""
    B = backends.active if backend is None else backends[backend]
    wbar = B.mean(fw, axis=-1)

    fct_diff = (
        B.abs(B.expand_dims(fct, -2) - B.expand_dims(fct, -1)) ** p
    )  # (... M D D)
    obs_diff = B.abs(B.expand_dims(obs, -2) - B.expand_dims(obs, -1)) ** p  # (... D D)

    E_1 = (fct_diff - B.expand_dims(obs_diff, axis=-3)) ** 2  # (... M D D)
    E_1 = B.sum(B.expand_dims(w, -3) * E_1, axis=(-2, -1))  # (... M)
    E_1 = B.mean(E_1 * fw * B.expand_dims(ow, axis=-1), axis=-1)  # (...)

    E_2 = (
        B.expand_dims(fct_diff, -3) - B.expand_dims(fct_diff, -4)
    ) ** 2  # (... M M D D)
    E_2 = B.sum(B.expand_dims(w, (-3, -4)) * E_2, axis=(-2, -1))  # (... M M)

    fw_prod = B.expand_dims(fw, axis=-2) * B.expand_dims(fw, axis=-1)  # (... M M)
    E_2 = B.mean(E_2 * fw_prod, axis=(-2, -1))  # (...)

    E_3 = B.sum(B.expand_dims(w, -3) * fct_diff**2, axis=(-2, -1))  # (... M)
    E_3 = B.mean(E_3 * fw, axis=-1)  # (...)
    E_3 -= B.sum(w * obs_diff**2, axis=(-2, -1)) * ow  # (...)
    E_3 *= wbar - ow  # (...)

    return E_1 - 0.5 * E_2 + E_3
