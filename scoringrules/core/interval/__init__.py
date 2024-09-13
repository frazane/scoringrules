try:
    from ._gufunc import _interval_score_gufunc, _weighted_interval_score_gufunc
except ImportError:
    _interval_score_gufunc = None
    _weighted_interval_score_gufunc = None

from ._score import _interval_score, _weighted_interval_score

__all__ = [
    "_interval_score",
    "_weighted_interval_score",
    "_interval_score_gufunc",
    "_weighted_interval_score_gufunc",
]
