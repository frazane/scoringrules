try:
    from ._gufunc import _error_spread_score_gufunc as _ess_gufunc
except ImportError:
    _ess_gufunc = None

from ._score import error_spread_score as ess

__all__ = ["ess", "_ess_gufunc"]
