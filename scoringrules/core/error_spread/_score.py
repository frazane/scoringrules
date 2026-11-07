import typing as tp

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array


def error_spread_score(obs: "Array", fct: "Array", *, xp) -> "Array":
    """Error spread score."""
    M: int = fct.shape[-1]
    # keepdims: for 1-d fct the reductions would give numpy scalars, which the
    # namespace's scalar-tolerant sqrt turns into Python floats
    m = xp.mean(fct, axis=-1, keepdims=True)
    s = xp.sqrt(xp.sum((fct - m) ** 2, axis=-1, keepdims=True) / (M - 1))
    g = xp.sum(((fct - m) / s) ** 3, axis=-1) / ((M - 1) * (M - 2))
    m, s = m[..., 0], s[..., 0]
    e = m - obs
    return (s**2 - e**2 - e * s * g) ** 2
