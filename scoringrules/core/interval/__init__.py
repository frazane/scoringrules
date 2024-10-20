try:
    from ._gufunc import _weighted_interval_score_gufunc
except ImportError:
    _weighted_interval_score_gufunc = None

from ._score import interval_score, weighted_interval_score

__all__ = [
    "interval_score",
    "weighted_interval_score",
    "_weighted_interval_score_gufunc",
]
