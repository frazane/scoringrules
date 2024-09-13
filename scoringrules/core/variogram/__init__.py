try:
    from ._gufuncs import (
        _owvariogram_score_gufunc,
        _variogram_score_gufunc,
        _vrvariogram_score_gufunc,
    )
except ImportError:
    _owvariogram_score_gufunc = None
    _variogram_score_gufunc = None
    _vrvariogram_score_gufunc = None

from ._score import owvariogram_score as owvs
from ._score import variogram_score as vs
from ._score import vrvariogram_score as vrvs

__all__ = [
    "vs",
    "owvs",
    "vrvs",
    "_variogram_score_gufunc",
    "_owvariogram_score_gufunc",
    "_vrvariogram_score_gufunc",
]
