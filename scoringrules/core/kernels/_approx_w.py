import typing as tp

from scoringrules.backend import backends

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike, Backend


def gauss_kern_uv(
    x1: "ArrayLike", x2: "ArrayLike", backend: "Backend" = None
) -> "Array":
    """Compute the gaussian kernel evaluated at x1 and x2."""
    B = backends.active if backend is None else backends[backend]
    return B.exp(-0.5 * (x1 - x2) ** 2)


def gauss_kern_mv(
    x1: "ArrayLike", x2: "ArrayLike", backend: "Backend" = None
) -> "Array":
    """Compute the gaussian kernel evaluated at vectors x1 and x2."""
    B = backends.active if backend is None else backends[backend]
    return B.exp(-0.5 * B.norm(x1 - x2, -1) ** 2)


def ensemble_uv_w(
    obs: "ArrayLike",
    fct: "Array",
    ens_w: "Array",
    estimator: str = "nrg",
    backend: "Backend" = None,
) -> "Array":
    """Compute a kernel score for a finite ensemble."""
    B = backends.active if backend is None else backends[backend]

    e_1 = B.sum(gauss_kern_uv(obs[..., None], fct, backend=backend) * ens_w, axis=-1)
    e_2 = B.sum(
        gauss_kern_uv(fct[..., None], fct[..., None, :], backend=backend)
        * ens_w[..., None]
        * ens_w[..., None, :],
        axis=(-1, -2),
    )
    e_3 = gauss_kern_uv(obs, obs)

    if estimator == "fair":
        e_2 = e_2 / (1 - B.sum(ens_w * ens_w, axis=-1))

    out = e_1 - 0.5 * e_2 - 0.5 * e_3
    return -out


def ensemble_mv_w(
    obs: "ArrayLike",
    fct: "Array",
    ens_w: "Array",
    estimator: str = "nrg",
    backend: "Backend" = None,
) -> "Array":
    """Compute a kernel score for a finite multivariate ensemble.

    The ensemble and variables axes are on the second last and last dimensions respectively.
    """
    B = backends.active if backend is None else backends[backend]

    e_1 = B.sum(
        ens_w * gauss_kern_mv(B.expand_dims(obs, -2), fct, backend=backend), axis=-1
    )
    e_2 = B.sum(
        gauss_kern_mv(B.expand_dims(fct, -3), B.expand_dims(fct, -2), backend=backend)
        * B.expand_dims(ens_w, -1)
        * B.expand_dims(ens_w, -2),
        axis=(-2, -1),
    )
    e_3 = gauss_kern_mv(obs, obs)

    if estimator == "fair":
        e_2 = e_2 / (1 - B.sum(ens_w * ens_w, axis=-1))

    out = e_1 - 0.5 * e_2 - 0.5 * e_3
    return -out


def ow_ensemble_uv_w(
    obs: "ArrayLike",
    fct: "Array",
    ow: "Array",
    fw: "Array",
    ens_w: "Array",
    backend: "Backend" = None,
) -> "Array":
    """Compute an outcome-weighted kernel score for a finite univariate ensemble."""
    B = backends.active if backend is None else backends[backend]
    wbar = B.sum(ens_w * fw, axis=-1)
    e_1 = (
        B.sum(ens_w * gauss_kern_uv(obs[..., None], fct, backend=backend) * fw, axis=-1)
        * ow
        / wbar
    )
    e_2 = B.sum(
        ens_w[..., None]
        * ens_w[..., None, :]
        * gauss_kern_uv(fct[..., None], fct[..., None, :], backend=backend)
        * fw[..., None]
        * fw[..., None, :],
        axis=(-1, -2),
    )
    e_2 *= ow / (wbar**2)
    e_3 = gauss_kern_uv(obs, obs, backend=backend) * ow

    out = e_1 - 0.5 * e_2 - 0.5 * e_3
    return -out


def ow_ensemble_mv_w(
    obs: "ArrayLike",
    fct: "Array",
    ow: "Array",
    fw: "Array",
    ens_w: "Array",
    backend: "Backend" = None,
) -> "Array":
    """Compute an outcome-weighted kernel score for a finite multivariate ensemble.

    The ensemble and variables axes are on the second last and last dimensions respectively.
    """
    B = backends.active if backend is None else backends[backend]
    wbar = B.sum(fw * ens_w, -1)

    err_kern = gauss_kern_mv(B.expand_dims(obs, -2), fct, backend=backend)
    E_1 = B.sum(err_kern * fw * B.expand_dims(ow, -1) * ens_w, axis=-1) / wbar

    spread_kern = gauss_kern_mv(
        B.expand_dims(fct, -3), B.expand_dims(fct, -2), backend=backend
    )
    fw_prod = B.expand_dims(fw, -1) * B.expand_dims(fw, -2)
    spread_kern *= fw_prod * B.expand_dims(ow, (-2, -1))
    E_2 = B.sum(
        spread_kern * B.expand_dims(ens_w, -1) * B.expand_dims(ens_w, -2), (-2, -1)
    ) / (wbar**2)

    E_3 = gauss_kern_mv(obs, obs, backend=backend) * ow

    out = E_1 - 0.5 * E_2 - 0.5 * E_3
    out = -out
    return out


def vr_ensemble_uv_w(
    obs: "ArrayLike",
    fct: "Array",
    ow: "Array",
    fw: "Array",
    ens_w: "Array",
    backend: "Backend" = None,
) -> "Array":
    """Compute a vertically re-scaled kernel score for a finite univariate ensemble."""
    B = backends.active if backend is None else backends[backend]

    e_1 = (
        B.sum(ens_w * gauss_kern_uv(obs[..., None], fct, backend=backend) * fw, axis=-1)
        * ow
    )
    e_2 = B.sum(
        ens_w[..., None]
        * ens_w[..., None, :]
        * gauss_kern_uv(fct[..., None], fct[..., None, :], backend=backend)
        * fw[..., None]
        * fw[..., None, :],
        axis=(-1, -2),
    )
    e_3 = gauss_kern_uv(obs, obs, backend=backend) * ow * ow

    out = e_1 - 0.5 * e_2 - 0.5 * e_3
    out = -out
    return out


def vr_ensemble_mv_w(
    obs: "ArrayLike",
    fct: "Array",
    ow: "Array",
    fw: "Array",
    ens_w: "Array",
    backend: "Backend" = None,
) -> "Array":
    """Compute a vertically re-scaled kernel score for a finite multivariate ensemble.

    The ensemble and variables axes are on the second last and last dimensions respectively.
    """
    B = backends.active if backend is None else backends[backend]

    err_kern = gauss_kern_mv(B.expand_dims(obs, -2), fct, backend=backend)
    E_1 = B.sum(err_kern * fw * B.expand_dims(ow, -1) * ens_w, axis=-1)

    spread_kern = gauss_kern_mv(
        B.expand_dims(fct, -3), B.expand_dims(fct, -2), backend=backend
    )
    fw_prod = B.expand_dims(fw, -1) * B.expand_dims(fw, -2)
    spread_kern *= fw_prod
    E_2 = B.sum(
        spread_kern * B.expand_dims(ens_w, -1) * B.expand_dims(ens_w, -2), (-2, -1)
    )

    E_3 = gauss_kern_mv(obs, obs, backend=backend) * ow * ow

    out = E_1 - 0.5 * E_2 - 0.5 * E_3
    out = -out
    return out
