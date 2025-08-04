try:
    from ._gufuncs import _dss_gufunc
except ImportError:
    _dss_gufunc = None

from ._score import ds_score as dss

__all__ = ["dss", "_dss_gufunc"]
