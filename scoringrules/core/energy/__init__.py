try:
    from ._gufuncs import (
        _energy_score_nrg_gufunc,
        _energy_score_fair_gufunc,
        _owenergy_score_gufunc,
        _vrenergy_score_gufunc,
    )
except ImportError:
    _energy_score_nrg_gufunc = None
    _energy_score_fair_gufunc = None
    _owenergy_score_gufunc = None
    _vrenergy_score_gufunc = None

from ._score import es_ensemble as es
from ._score import owes_ensemble as owes
from ._score import vres_ensemble as vres

__all__ = [
    "es",
    "owes",
    "vres",
    "_energy_score_nrg_gufunc",
    "_energy_score_fair_gufunc",
    "_owenergy_score_gufunc",
    "_vrenergy_score_gufunc",
]
