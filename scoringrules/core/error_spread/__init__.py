from scoringrules.core.utils import lazy_gufunc_getattr

# numba gufuncs, imported on first access (``import scoringrules`` must not import numba)
__getattr__ = lazy_gufunc_getattr(
    __name__,
    {"_ess_gufunc": "._gufunc:_error_spread_score_gufunc"},
)

from ._score import error_spread_score as ess

__all__ = ["ess", "_ess_gufunc"]
