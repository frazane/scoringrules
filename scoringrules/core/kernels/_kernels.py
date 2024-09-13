import typing as tp

from scoringrules.backend import backends

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike, Backend


def gauss_kern(x1: "ArrayLike", x2: "ArrayLike", backend: "Backend" = None) -> "Array":
    """Compute the gaussian kernel evaluated at x1 and x2."""
    B = backends.active if backend is None else backends[backend]
    return B.exp(-0.5 * (x1 - x2) ** 2)
