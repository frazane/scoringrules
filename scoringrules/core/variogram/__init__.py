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

from ._score import owvs_ensemble as owvs
from ._score import vs_ensemble as vs
from ._score import vrvs_ensemble as vrvs

__all__ = [
    "vs",
    "owvs",
    "vrvs",
    "_variogram_score_gufunc",
    "_owvariogram_score_gufunc",
    "_vrvariogram_score_gufunc",
]
