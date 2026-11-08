from scoringrules.core.utils import lazy_gufunc_getattr

# numba gufuncs, imported on first access (``import scoringrules`` must not import numba)
__getattr__ = lazy_gufunc_getattr(
    __name__,
    {
        "_weighted_interval_score_gufunc": "._gufunc",
    },
)

from ._score import interval_score, weighted_interval_score

__all__ = [
    "interval_score",
    "weighted_interval_score",
    "_weighted_interval_score_gufunc",
]
