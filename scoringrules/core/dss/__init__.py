from scoringrules.core.utils import lazy_gufunc_getattr

# numba gufuncs, imported on first access (``import scoringrules`` must not import numba)
__getattr__ = lazy_gufunc_getattr(
    __name__,
    {
        "_dss_uv_gufunc": "._gufuncs",
        "_dss_mv_gufunc": "._gufuncs",
    },
)

from ._score import ds_score_uv, ds_score_mv

__all__ = ["ds_score_uv", "_dss_uv_gufunc", "ds_score_mv", "_dss_mv_gufunc"]
