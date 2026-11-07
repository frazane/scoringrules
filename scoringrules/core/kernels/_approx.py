import typing as tp

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike


def gauss_kern_uv(x1: "ArrayLike", x2: "ArrayLike", *, xp) -> "Array":
    """Compute the gaussian kernel evaluated at x1 and x2."""
    return xp.exp(-0.5 * (x1 - x2) ** 2)


def gauss_kern_mv(x1: "ArrayLike", x2: "ArrayLike", *, xp) -> "Array":
    """Compute the gaussian kernel evaluated at vectors x1 and x2."""
    # squared distance as a sum of squares, not norm ** 2: the norm has no
    # derivative at 0 (the i == j pairs and k(obs, obs)), so jax grads were NaN
    d = x1 - x2
    return xp.exp(-0.5 * xp.sum(d * d, axis=-1))


def ensemble_uv(
    obs: "ArrayLike",
    fct: "Array",
    estimator: str = "nrg",
    *,
    xp,
) -> "Array":
    """Compute a kernel score for a finite ensemble."""
    if estimator == "nrg":
        out = _ks_ensemble_nrg_uv(obs, fct, xp=xp)
    elif estimator == "fair":
        out = _ks_ensemble_fair_uv(obs, fct, xp=xp)
    elif estimator == "akr":
        out = _ks_ensemble_akr_uv(obs, fct, xp=xp)
    elif estimator == "akr_circperm":
        out = _ks_ensemble_akr_circperm_uv(obs, fct, xp=xp)
    else:
        raise ValueError(
            f"{estimator} not a valid estimator, must be one of 'nrg', 'fair', 'akr', and 'akr_circperm'."
        )

    return out


def _ks_ensemble_nrg_uv(obs: "ArrayLike", fct: "Array", *, xp) -> "Array":
    """Standard version of the kernel score."""
    M: int = fct.shape[-1]
    e_1 = xp.sum(gauss_kern_uv(obs[..., None], fct, xp=xp), axis=-1)
    e_2 = xp.sum(
        gauss_kern_uv(fct[..., None], fct[..., None, :], xp=xp),
        axis=(-1, -2),
    )
    e_3 = gauss_kern_uv(obs, obs, xp=xp)

    out = e_1 / M - 0.5 * e_2 / (M**2) - 0.5 * e_3

    return -out


def _ks_ensemble_fair_uv(obs: "ArrayLike", fct: "Array", *, xp) -> "Array":
    """Fair version of the kernel score."""
    M: int = fct.shape[-1]
    e_1 = xp.sum(gauss_kern_uv(obs[..., None], fct, xp=xp), axis=-1)
    e_2 = xp.sum(
        gauss_kern_uv(fct[..., None], fct[..., None, :], xp=xp),
        axis=(-1, -2),
    )
    e_2 = e_2 - xp.sum(gauss_kern_uv(fct, fct, xp=xp), axis=-1)
    e_3 = gauss_kern_uv(obs, obs, xp=xp)

    out = e_1 / M - 0.5 * e_2 / (M * (M - 1)) - 0.5 * e_3

    return -out


def _ks_ensemble_akr_uv(obs: "ArrayLike", fct: "Array", *, xp) -> "Array":
    """Approximate kernel representation estimator of the kernel score."""
    M: int = fct.shape[-1]
    e_1 = xp.sum(gauss_kern_uv(obs[..., None], fct, xp=xp), axis=-1)
    e_2 = xp.sum(gauss_kern_uv(fct, xp.roll(fct, shift=1, axis=-1), xp=xp), axis=-1)
    e_3 = gauss_kern_uv(obs, obs, xp=xp)

    out = e_1 / M - 0.5 * e_2 / M - 0.5 * e_3

    return -out


def _ks_ensemble_akr_circperm_uv(obs: "ArrayLike", fct: "Array", *, xp) -> "Array":
    """AKR estimator of the kernel score with cyclic permutation."""
    M: int = fct.shape[-1]
    e_1 = xp.sum(gauss_kern_uv(obs[..., None], fct, xp=xp), axis=-1)
    # member i is paired with member (i + M // 2) % M
    e_2 = xp.sum(
        gauss_kern_uv(fct, xp.roll(fct, shift=-(M // 2), axis=-1), xp=xp),
        axis=-1,
    )
    e_3 = gauss_kern_uv(obs, obs, xp=xp)

    out = e_1 / M - 0.5 * e_2 / M - 0.5 * e_3

    return -out


def ensemble_mv(
    obs: "ArrayLike",
    fct: "Array",
    estimator: str = "nrg",
    *,
    xp,
) -> "Array":
    """Compute a kernel score for a finite multivariate ensemble.

    The ensemble and variables axes are on the second last and last dimensions respectively.
    """
    if estimator == "nrg":
        out = _ks_ensemble_nrg_mv(obs, fct, xp=xp)
    elif estimator == "fair":
        out = _ks_ensemble_fair_mv(obs, fct, xp=xp)
    elif estimator == "akr":
        out = _ks_ensemble_akr_mv(obs, fct, xp=xp)
    elif estimator == "akr_circperm":
        out = _ks_ensemble_akr_circperm_mv(obs, fct, xp=xp)
    else:
        raise ValueError(
            f"{estimator} not a valid estimator, must be one of 'nrg', 'fair', 'akr', and 'akr_circperm'."
        )

    return out


def _ks_ensemble_nrg_mv(obs: "ArrayLike", fct: "Array", *, xp) -> "Array":
    M: int = fct.shape[-2]

    e_1 = xp.sum(gauss_kern_mv(xp.expand_dims(obs, axis=-2), fct, xp=xp), axis=-1)
    e_2 = xp.sum(
        gauss_kern_mv(
            xp.expand_dims(fct, axis=-3), xp.expand_dims(fct, axis=-2), xp=xp
        ),
        axis=(-2, -1),
    )
    e_3 = gauss_kern_mv(obs, obs, xp=xp)

    out = e_1 / M - 0.5 * e_2 / (M**2) - 0.5 * e_3

    return -out


def _ks_ensemble_fair_mv(obs: "ArrayLike", fct: "Array", *, xp) -> "Array":
    M: int = fct.shape[-2]

    e_1 = xp.sum(gauss_kern_mv(xp.expand_dims(obs, axis=-2), fct, xp=xp), axis=-1)
    e_2 = xp.sum(
        gauss_kern_mv(
            xp.expand_dims(fct, axis=-3), xp.expand_dims(fct, axis=-2), xp=xp
        ),
        axis=(-2, -1),
    )
    e_2 = e_2 - xp.sum(gauss_kern_mv(fct, fct, xp=xp), axis=-1)
    e_3 = gauss_kern_mv(obs, obs, xp=xp)

    out = e_1 / M - 0.5 * e_2 / (M * (M - 1)) - 0.5 * e_3

    return -out


def _ks_ensemble_akr_mv(obs: "ArrayLike", fct: "Array", *, xp) -> "Array":
    M: int = fct.shape[-2]

    e_1 = xp.sum(gauss_kern_mv(xp.expand_dims(obs, axis=-2), fct, xp=xp), axis=-1)
    e_2 = xp.sum(gauss_kern_mv(fct, xp.roll(fct, shift=1, axis=-2), xp=xp), axis=-1)
    e_3 = gauss_kern_mv(obs, obs, xp=xp)

    out = e_1 / M - 0.5 * e_2 / M - 0.5 * e_3

    return -out


def _ks_ensemble_akr_circperm_mv(obs: "ArrayLike", fct: "Array", *, xp) -> "Array":
    M: int = fct.shape[-2]

    e_1 = xp.sum(gauss_kern_mv(xp.expand_dims(obs, axis=-2), fct, xp=xp), axis=-1)
    # member i is paired with member (i + M // 2) % M
    e_2 = xp.sum(
        gauss_kern_mv(fct, xp.roll(fct, shift=-(M // 2), axis=-2), xp=xp),
        axis=-1,
    )
    e_3 = gauss_kern_mv(obs, obs, xp=xp)

    out = e_1 / M - 0.5 * e_2 / M - 0.5 * e_3

    return -out


def ow_ensemble_uv(
    obs: "ArrayLike",
    fct: "Array",
    ow: "Array",
    fw: "Array",
    *,
    xp,
) -> "Array":
    """Compute an outcome-weighted kernel score for a finite univariate ensemble."""
    M: int = fct.shape[-1]
    # sum / M, not mean: mean rejects a boolean fw on torch
    wbar = xp.sum(fw, axis=-1) / M
    e_1 = (
        xp.sum(gauss_kern_uv(obs[..., None], fct, xp=xp) * fw, axis=-1)
        * ow
        / (M * wbar)
    )
    e_2 = xp.sum(
        gauss_kern_uv(fct[..., None], fct[..., None, :], xp=xp)
        * fw[..., None]
        * fw[..., None, :],
        axis=(-1, -2),
    )
    e_2 = e_2 * (ow / (M**2 * wbar**2))
    e_3 = gauss_kern_uv(obs, obs, xp=xp) * ow

    out = e_1 - 0.5 * e_2 - 0.5 * e_3
    return -out


def ow_ensemble_mv(
    obs: "ArrayLike",
    fct: "Array",
    ow: "Array",
    fw: "Array",
    *,
    xp,
) -> "Array":
    """Compute an outcome-weighted kernel score for a finite multivariate ensemble.

    The ensemble and variables axes are on the second last and last dimensions respectively.
    """
    M: int = fct.shape[-2]
    wbar = xp.sum(fw, axis=-1) / M

    err_kern = gauss_kern_mv(xp.expand_dims(obs, axis=-2), fct, xp=xp)
    E_1 = xp.sum(err_kern * fw * xp.expand_dims(ow, axis=-1), axis=-1) / (M * wbar)

    spread_kern = gauss_kern_mv(
        xp.expand_dims(fct, axis=-3), xp.expand_dims(fct, axis=-2), xp=xp
    )
    fw_prod = xp.expand_dims(fw, axis=-1) * xp.expand_dims(fw, axis=-2)
    # out of place: torch autograd needs the kernel output unchanged
    spread_kern = spread_kern * (fw_prod * ow[..., None, None])
    E_2 = xp.sum(spread_kern, axis=(-2, -1)) / (M**2 * wbar**2)

    E_3 = gauss_kern_mv(obs, obs, xp=xp) * ow

    out = E_1 - 0.5 * E_2 - 0.5 * E_3
    out = -out
    return out


def vr_ensemble_uv(
    obs: "ArrayLike",
    fct: "Array",
    ow: "Array",
    fw: "Array",
    *,
    xp,
) -> "Array":
    """Compute a vertically re-scaled kernel score for a finite univariate ensemble."""
    M: int = fct.shape[-1]

    e_1 = xp.sum(gauss_kern_uv(obs[..., None], fct, xp=xp) * fw, axis=-1) * ow / M
    e_2 = xp.sum(
        gauss_kern_uv(xp.expand_dims(fct, axis=-1), xp.expand_dims(fct, axis=-2), xp=xp)
        * (xp.expand_dims(fw, axis=-1) * xp.expand_dims(fw, axis=-2)),
        axis=(-1, -2),
    ) / (M**2)
    e_3 = gauss_kern_uv(obs, obs, xp=xp) * ow * ow

    out = e_1 - 0.5 * e_2 - 0.5 * e_3
    out = -out
    return out


def vr_ensemble_mv(
    obs: "ArrayLike",
    fct: "Array",
    ow: "Array",
    fw: "Array",
    *,
    xp,
) -> "Array":
    """Compute a vertically re-scaled kernel score for a finite multivariate ensemble.

    The ensemble and variables axes are on the second last and last dimensions respectively.
    """
    M: int = fct.shape[-2]

    err_kern = gauss_kern_mv(xp.expand_dims(obs, axis=-2), fct, xp=xp)
    E_1 = xp.sum(err_kern * fw * xp.expand_dims(ow, axis=-1), axis=-1) / M

    spread_kern = gauss_kern_mv(
        xp.expand_dims(fct, axis=-3), xp.expand_dims(fct, axis=-2), xp=xp
    )
    fw_prod = xp.expand_dims(fw, axis=-1) * xp.expand_dims(fw, axis=-2)
    spread_kern = spread_kern * fw_prod
    E_2 = xp.sum(spread_kern, axis=(-2, -1)) / (M**2)

    E_3 = gauss_kern_mv(obs, obs, xp=xp) * ow * ow

    out = E_1 - 0.5 * E_2 - 0.5 * E_3
    out = -out
    return out
