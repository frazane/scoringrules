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

from ._score import es_ensemble as nrg
from ._score import owes_ensemble as ownrg
from ._score import vres_ensemble as vrnrg

__all__ = [
    "nrg",
    "ownrg",
    "vrnrg",
    "_energy_score_gufunc",
    "_owenergy_score_gufunc",
    "_vrenergy_score_gufunc",
]
