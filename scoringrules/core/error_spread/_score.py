import typing as tp

from scoringrules.backend import backends

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, Backend


def error_spread_score(
    fcts: "Array", obs: "Array", backend: "Backend" = None
) -> "Array":
    """Error spread score."""
    B = backends.active if backend is None else backends[backend]
    M: int = fcts.shape[-1]
    m = B.mean(fcts, axis=-1)
    s = B.sqrt(B.sum((fcts - m[...,None]) ** 2, axis=-1) / (M - 1))
    g = B.sum(((fcts - m[...,None]) / s[...,None]) ** 3, axis=-1) / ((M - 1) * (M - 2))
    e = m - obs
    return B.squeeze((s**2 - e**2 - e * s * g) ** 2)


