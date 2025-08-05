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

try:
    from ._gufuncs_w import (
        _owvariogram_score_gufunc_w,
        _variogram_score_gufunc_w,
        _vrvariogram_score_gufunc_w,
    )
except ImportError:
    _owvariogram_score_gufunc_w = None
    _variogram_score_gufunc_w = None
    _vrvariogram_score_gufunc_w = None

from ._score import owvs_ensemble as owvs
from ._score import vs_ensemble as vs
from ._score import vrvs_ensemble as vrvs

from ._score_w import owvs_ensemble_w as owvs_w
from ._score_w import vs_ensemble_w as vs_w
from ._score_w import vrvs_ensemble_w as vrvs_w

__all__ = [
    "vs",
    "vs_w",
    "owvs",
    "owvs_w",
    "vrvs",
    "vrvs_w",
    "_variogram_score_gufunc",
    "_variogram_score_gufunc_w",
    "_owvariogram_score_gufunc",
    "_owvariogram_score_gufunc_w",
    "_vrvariogram_score_gufunc",
    "_vrvariogram_score_gufunc_w",
]
