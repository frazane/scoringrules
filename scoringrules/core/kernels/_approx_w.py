import typing as tp

from ._approx import gauss_kern_mv, gauss_kern_uv

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike


def ensemble_uv_w(
    obs: "ArrayLike",
    fct: "Array",
    ens_w: "Array",
    estimator: str = "nrg",
    *,
    xp,
) -> "Array":
    """Compute a kernel score for a finite ensemble."""
    if estimator == "nrg":
        out = _ks_ensemble_nrg_uv_w(obs, fct, ens_w, xp=xp)
    elif estimator == "fair":
        out = _ks_ensemble_fair_uv_w(obs, fct, ens_w, xp=xp)
    elif estimator == "akr":
        out = _ks_ensemble_akr_uv_w(obs, fct, ens_w, xp=xp)
    elif estimator == "akr_circperm":
        out = _ks_ensemble_akr_circperm_uv_w(obs, fct, ens_w, xp=xp)
    else:
        raise ValueError(
            f"{estimator} not a valid estimator, must be one of 'nrg', 'fair', 'akr', and 'akr_circperm'."
        )

    return out


def _ks_ensemble_nrg_uv_w(
    obs: "ArrayLike", fct: "Array", ens_w: "Array", *, xp
) -> "Array":
    """Energy estimator of the kernel score."""
    e_1 = xp.sum(gauss_kern_uv(obs[..., None], fct, xp=xp) * ens_w, axis=-1)
    e_2 = xp.sum(
        gauss_kern_uv(fct[..., None], fct[..., None, :], xp=xp)
        * ens_w[..., None]
        * ens_w[..., None, :],
        axis=(-1, -2),
    )
    e_3 = gauss_kern_uv(obs, obs, xp=xp)

    out = e_1 - 0.5 * e_2 - 0.5 * e_3
    return -out


def _ks_ensemble_fair_uv_w(
    obs: "ArrayLike", fct: "Array", ens_w: "Array", *, xp
) -> "Array":
    """Fair estimator of the kernel score."""
    e_1 = xp.sum(gauss_kern_uv(obs[..., None], fct, xp=xp) * ens_w, axis=-1)
    e_2 = xp.sum(
        gauss_kern_uv(fct[..., None], fct[..., None, :], xp=xp)
        * ens_w[..., None]
        * ens_w[..., None, :],
        axis=(-1, -2),
    )
    e_2 = e_2 - xp.sum(gauss_kern_uv(fct, fct, xp=xp) * ens_w * ens_w, axis=-1)
    e_3 = gauss_kern_uv(obs, obs, xp=xp)

    fair_c = 1 - xp.sum(ens_w * ens_w, axis=-1)

    out = e_1 - 0.5 * e_2 / fair_c - 0.5 * e_3
    return -out


def _ks_ensemble_akr_uv_w(
    obs: "ArrayLike", fct: "Array", ens_w: "Array", *, xp
) -> "Array":
    """Approximate kernel representation estimator of the kernel score."""
    e_1 = xp.sum(gauss_kern_uv(obs[..., None], fct, xp=xp) * ens_w, axis=-1)
    e_2 = xp.sum(
        gauss_kern_uv(fct, xp.roll(fct, shift=1, axis=-1), xp=xp) * ens_w,
        axis=-1,
    )
    e_3 = gauss_kern_uv(obs, obs, xp=xp)

    out = e_1 - 0.5 * e_2 - 0.5 * e_3

    return -out


def _ks_ensemble_akr_circperm_uv_w(
    obs: "ArrayLike", fct: "Array", ens_w: "Array", *, xp
) -> "Array":
    """AKR estimator of the kernel score with cyclic permutation."""
    M: int = fct.shape[-1]
    e_1 = xp.sum(gauss_kern_uv(obs[..., None], fct, xp=xp) * ens_w, axis=-1)
    # member i is paired with member (i + M // 2) % M
    e_2 = xp.sum(
        gauss_kern_uv(fct, xp.roll(fct, shift=-(M // 2), axis=-1), xp=xp) * ens_w,
        axis=-1,
    )
    e_3 = gauss_kern_uv(obs, obs, xp=xp)

    out = e_1 - 0.5 * e_2 - 0.5 * e_3

    return -out


def ensemble_mv_w(
    obs: "ArrayLike",
    fct: "Array",
    ens_w: "Array",
    estimator: str = "nrg",
    *,
    xp,
) -> "Array":
    """Compute a kernel score for a finite multivariate ensemble."""
    if estimator == "nrg":
        out = _ks_ensemble_nrg_mv_w(obs, fct, ens_w, xp=xp)
    elif estimator == "fair":
        out = _ks_ensemble_fair_mv_w(obs, fct, ens_w, xp=xp)
    elif estimator == "akr":
        out = _ks_ensemble_akr_mv_w(obs, fct, ens_w, xp=xp)
    elif estimator == "akr_circperm":
        out = _ks_ensemble_akr_circperm_mv_w(obs, fct, ens_w, xp=xp)
    else:
        raise ValueError(
            f"{estimator} not a valid estimator, must be one of 'nrg', 'fair', 'akr', and 'akr_circperm'."
        )

    return out


def _ks_ensemble_nrg_mv_w(
    obs: "ArrayLike", fct: "Array", ens_w: "Array", *, xp
) -> "Array":
    """Energy estimator of the kernel score."""
    e_1 = xp.sum(
        ens_w * gauss_kern_mv(xp.expand_dims(obs, axis=-2), fct, xp=xp), axis=-1
    )
    e_2 = xp.sum(
        gauss_kern_mv(xp.expand_dims(fct, axis=-3), xp.expand_dims(fct, axis=-2), xp=xp)
        * xp.expand_dims(ens_w, axis=-1)
        * xp.expand_dims(ens_w, axis=-2),
        axis=(-2, -1),
    )
    e_3 = gauss_kern_mv(obs, obs, xp=xp)

    out = e_1 - 0.5 * e_2 - 0.5 * e_3
    return -out


def _ks_ensemble_fair_mv_w(
    obs: "ArrayLike", fct: "Array", ens_w: "Array", *, xp
) -> "Array":
    """Fair estimator of the kernel score."""
    e_1 = xp.sum(
        ens_w * gauss_kern_mv(xp.expand_dims(obs, axis=-2), fct, xp=xp), axis=-1
    )
    e_2 = xp.sum(
        gauss_kern_mv(xp.expand_dims(fct, axis=-3), xp.expand_dims(fct, axis=-2), xp=xp)
        * xp.expand_dims(ens_w, axis=-1)
        * xp.expand_dims(ens_w, axis=-2),
        axis=(-2, -1),
    )
    e_2 = e_2 - xp.sum(gauss_kern_mv(fct, fct, xp=xp) * ens_w * ens_w, axis=-1)
    e_3 = gauss_kern_mv(obs, obs, xp=xp)

    fair_c = 1 - xp.sum(ens_w * ens_w, axis=-1)

    out = e_1 - 0.5 * e_2 / fair_c - 0.5 * e_3
    return -out


def _ks_ensemble_akr_mv_w(
    obs: "ArrayLike", fct: "Array", ens_w: "Array", *, xp
) -> "Array":
    e_1 = xp.sum(
        gauss_kern_mv(xp.expand_dims(obs, axis=-2), fct, xp=xp) * ens_w, axis=-1
    )
    e_2 = xp.sum(
        gauss_kern_mv(fct, xp.roll(fct, shift=1, axis=-2), xp=xp) * ens_w,
        axis=-1,
    )
    e_3 = gauss_kern_mv(obs, obs, xp=xp)

    out = e_1 - 0.5 * e_2 - 0.5 * e_3

    return -out


def _ks_ensemble_akr_circperm_mv_w(
    obs: "ArrayLike", fct: "Array", ens_w: "Array", *, xp
) -> "Array":
    M: int = fct.shape[-2]

    e_1 = xp.sum(
        gauss_kern_mv(xp.expand_dims(obs, axis=-2), fct, xp=xp) * ens_w, axis=-1
    )
    # member i is paired with member (i + M // 2) % M
    e_2 = xp.sum(
        gauss_kern_mv(fct, xp.roll(fct, shift=-(M // 2), axis=-2), xp=xp) * ens_w,
        axis=-1,
    )
    e_3 = gauss_kern_mv(obs, obs, xp=xp)

    out = e_1 - 0.5 * e_2 - 0.5 * e_3

    return -out


def ow_ensemble_uv_w(
    obs: "ArrayLike",
    fct: "Array",
    ow: "Array",
    fw: "Array",
    ens_w: "Array",
    *,
    xp,
) -> "Array":
    """Compute an outcome-weighted kernel score for a finite univariate ensemble."""
    wbar = xp.sum(ens_w * fw, axis=-1)
    e_1 = (
        xp.sum(ens_w * gauss_kern_uv(obs[..., None], fct, xp=xp) * fw, axis=-1)
        * ow
        / wbar
    )
    e_2 = xp.sum(
        ens_w[..., None]
        * ens_w[..., None, :]
        * gauss_kern_uv(fct[..., None], fct[..., None, :], xp=xp)
        * fw[..., None]
        * fw[..., None, :],
        axis=(-1, -2),
    )
    e_2 = e_2 * (ow / (wbar**2))
    e_3 = gauss_kern_uv(obs, obs, xp=xp) * ow

    out = e_1 - 0.5 * e_2 - 0.5 * e_3
    return -out


def ow_ensemble_mv_w(
    obs: "ArrayLike",
    fct: "Array",
    ow: "Array",
    fw: "Array",
    ens_w: "Array",
    *,
    xp,
) -> "Array":
    """Compute an outcome-weighted kernel score for a finite multivariate ensemble.

    The ensemble and variables axes are on the second last and last dimensions respectively.
    """
    wbar = xp.sum(fw * ens_w, axis=-1)

    err_kern = gauss_kern_mv(xp.expand_dims(obs, axis=-2), fct, xp=xp)
    E_1 = xp.sum(err_kern * fw * xp.expand_dims(ow, axis=-1) * ens_w, axis=-1) / wbar

    spread_kern = gauss_kern_mv(
        xp.expand_dims(fct, axis=-3), xp.expand_dims(fct, axis=-2), xp=xp
    )
    fw_prod = xp.expand_dims(fw, axis=-1) * xp.expand_dims(fw, axis=-2)
    # out of place: torch autograd needs the kernel output unchanged
    spread_kern = spread_kern * (fw_prod * ow[..., None, None])
    E_2 = xp.sum(
        spread_kern * xp.expand_dims(ens_w, axis=-1) * xp.expand_dims(ens_w, axis=-2),
        axis=(-2, -1),
    ) / (wbar**2)

    E_3 = gauss_kern_mv(obs, obs, xp=xp) * ow

    out = E_1 - 0.5 * E_2 - 0.5 * E_3
    out = -out
    return out


def vr_ensemble_uv_w(
    obs: "ArrayLike",
    fct: "Array",
    ow: "Array",
    fw: "Array",
    ens_w: "Array",
    *,
    xp,
) -> "Array":
    """Compute a vertically re-scaled kernel score for a finite univariate ensemble."""
    e_1 = xp.sum(ens_w * gauss_kern_uv(obs[..., None], fct, xp=xp) * fw, axis=-1) * ow
    e_2 = xp.sum(
        ens_w[..., None]
        * ens_w[..., None, :]
        * gauss_kern_uv(fct[..., None], fct[..., None, :], xp=xp)
        * fw[..., None]
        * fw[..., None, :],
        axis=(-1, -2),
    )
    e_3 = gauss_kern_uv(obs, obs, xp=xp) * ow * ow

    out = e_1 - 0.5 * e_2 - 0.5 * e_3
    out = -out
    return out


def vr_ensemble_mv_w(
    obs: "ArrayLike",
    fct: "Array",
    ow: "Array",
    fw: "Array",
    ens_w: "Array",
    *,
    xp,
) -> "Array":
    """Compute a vertically re-scaled kernel score for a finite multivariate ensemble.

    The ensemble and variables axes are on the second last and last dimensions respectively.
    """
    err_kern = gauss_kern_mv(xp.expand_dims(obs, axis=-2), fct, xp=xp)
    E_1 = xp.sum(err_kern * fw * xp.expand_dims(ow, axis=-1) * ens_w, axis=-1)

    spread_kern = gauss_kern_mv(
        xp.expand_dims(fct, axis=-3), xp.expand_dims(fct, axis=-2), xp=xp
    )
    fw_prod = xp.expand_dims(fw, axis=-1) * xp.expand_dims(fw, axis=-2)
    spread_kern = spread_kern * fw_prod
    E_2 = xp.sum(
        spread_kern * xp.expand_dims(ens_w, axis=-1) * xp.expand_dims(ens_w, axis=-2),
        axis=(-2, -1),
    )

    E_3 = gauss_kern_mv(obs, obs, xp=xp) * ow * ow

    out = E_1 - 0.5 * E_2 - 0.5 * E_3
    out = -out
    return out
