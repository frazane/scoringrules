import typing as tp

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike


def quantile_score(
    obs: "ArrayLike", fct: "ArrayLike", alpha: "ArrayLike", *, xp
) -> "Array":
    """Compute the quantile (pinball) score at quantile level ``alpha``."""
    obs, fct, alpha = map(xp.asarray, (obs, fct, alpha))
    if xp.any(alpha < 0) or xp.any(alpha > 1):
        raise ValueError("The quantile level must be between 0 and 1.")
    dtype = xp.result_type(*(xp.float_dtype(x) for x in (obs, fct, alpha)))
    return (xp.astype(obs < fct, dtype) - alpha) * (fct - obs)
