try:
    from ._gufuncs import (
        _dss_gufunc,
    )
except ImportError:
    _dss_gufunc = None

from ._score import _ds_score as _dss

__all__ = ["_dss_gufunc", "_dss"]
