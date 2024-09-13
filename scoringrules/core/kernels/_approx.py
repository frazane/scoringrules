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


def ensemble_uv(
    obs: "ArrayLike",
    fct: "Array",
    estimator: str = "nrg",
    backend: "Backend" = None,
) -> "Array":
    """Compute a kernel score for a finite ensemble."""
    B = backends.active if backend is None else backends[backend]
    M: int = fct.shape[-1]
    e_1 = B.sum(gauss_kern_uv(obs[..., None], fct, backend=backend), axis=-1) / M
    e_2 = B.sum(
        gauss_kern_uv(fct[..., None], fct[..., None, :], backend=backend), axis=(-1, -2)
    ) / (M**2)
    e_3 = gauss_kern_uv(obs, obs)

    if estimator == "nrg":
        out = e_1 - 0.5 * e_2 - 0.5 * e_3
    elif estimator == "fair":
        out = e_1 - 0.5 * e_2 * (M / (M - 1)) - 0.5 * e_3
    else:
        raise ValueError(f"{estimator} must be one of 'nrg' and 'fair'")

    out = -out
    return out


def ensemble_mv(
    obs: "ArrayLike",
    fct: "Array",
    estimator: str = "nrg",
    backend: "Backend" = None,
) -> "Array":
    """Compute a kernel score for a finite multivariate ensemble.

    The ensemble and variables axes are on the second last and last dimensions respectively.
    """
    B = backends.active if backend is None else backends[backend]
    M: int = fct.shape[-2]

    e_1 = (
        B.sum(gauss_kern_mv(B.expand_dims(obs, -2), fct, backend=backend), axis=-1) / M
    )
    e_2 = B.sum(
        gauss_kern_mv(B.expand_dims(fct, -3), B.expand_dims(fct, -2), backend=backend),
        axis=(-2, -1),
    ) / (M**2)
    e_3 = gauss_kern_mv(obs, obs)

    if estimator == "nrg":
        out = e_1 - 0.5 * e_2 - 0.5 * e_3
    elif estimator == "fair":
        out = e_1 - 0.5 * e_2 * (M / (M - 1)) - 0.5 * e_3
    else:
        raise ValueError(f"{estimator} must be one of 'nrg' and 'fair'")

    out = -out
    return out
