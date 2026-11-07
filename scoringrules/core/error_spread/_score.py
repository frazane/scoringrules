import typing as tp

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array


def error_spread_score(obs: "Array", fct: "Array", *, xp) -> "Array":
    """Error spread score."""
    M: int = fct.shape[-1]
    m = xp.mean(fct, axis=-1)
    s = xp.sqrt(xp.sum((fct - m[..., None]) ** 2, axis=-1) / (M - 1))
    g = xp.sum(((fct - m[..., None]) / s[..., None]) ** 3, axis=-1) / (
        (M - 1) * (M - 2)
    )
    e = m - obs
    return (s**2 - e**2 - e * s * g) ** 2
