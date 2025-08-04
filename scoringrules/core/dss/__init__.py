try:
    from ._gufuncs import _dss_mv_gufunc, _dss_uv_gufunc
except ImportError:
    _dss_uv_gufunc = None
    _dss_mv_gufunc = None

from ._score import ds_score_uv, ds_score_mv

__all__ = ["ds_score_uv", "_dss_uv_gufunc", "ds_score_mv", "_dss_mv_gufunc"]
