try:
    from ._gufuncs import (
        _energy_score_gufunc,
        _owenergy_score_gufunc,
        _vrenergy_score_gufunc,
    )
except ImportError:
    _energy_score_gufunc = None
    _owenergy_score_gufunc = None
    _vrenergy_score_gufunc = None

from ._score import energy_score as nrg
from ._score import owenergy_score as ownrg
from ._score import vrenergy_score as vrnrg

__all__ = [
    "nrg",
    "ownrg",
    "vrnrg",
    "_energy_score_gufunc",
    "_owenergy_score_gufunc",
    "_vrenergy_score_gufunc",
]
