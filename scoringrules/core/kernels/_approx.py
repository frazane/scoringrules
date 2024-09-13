import typing as tp

from scoringrules.backend import backends

from ._kernels import gauss_kern

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike, Backend


def ensemble(
    obs: "ArrayLike",
    fct: "Array",
    estimator: str = "nrg",
    backend: "Backend" = None,
) -> "Array":
    """Compute a kernel score for a finite ensemble."""
    B = backends.active if backend is None else backends[backend]
    M: int = fct.shape[-1]
    e_1 = B.sum(gauss_kern(obs[..., None], fct, backend=backend), axis=-1) / M
    e_2 = B.sum(
        gauss_kern(fct[..., None], fct[..., None, :], backend=backend), axis=(-1, -2)
    ) / (M**2)

    if estimator == "nrg":
        out = e_1 - 0.5 * e_2
    elif estimator == "fair":
        out = e_1 - 0.5 * e_2 * (M / (M - 1))
    else:
        raise ValueError(f"{estimator} must be one of 'nrg' and 'fair'")

    return out
