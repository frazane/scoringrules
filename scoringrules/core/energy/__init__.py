try:
    from ._gufuncs_w import (
        _energy_score_gufunc_w,
        _owenergy_score_gufunc_w,
        _vrenergy_score_gufunc_w,
    )

except ImportError:
    _energy_score_gufunc = None
    _energy_score_gufunc_w = None
    _owenergy_score_gufunc = None
    _owenergy_score_gufunc_w = None
    _vrenergy_score_gufunc = None
    _vrenergy_score_gufunc_w = None

from ._score import es_ensemble as nrg
from ._score import owes_ensemble as ownrg
from ._score import vres_ensemble as vrnrg

from ._score_w import es_ensemble_w as nrg_w
from ._score_w import owes_ensemble_w as ownrg_w
from ._score_w import vres_ensemble_w as vrnrg_w

__all__ = [
    "nrg",
    "nrg_w",
    "ownrg",
    "ownrg_w",
    "vrnrg",
    "vrnrg_w",
    "_energy_score_gufunc",
    "_energy_score_gufunc_w",
    "_owenergy_score_gufunc",
    "_owenergy_score_gufunc_w",
    "_vrenergy_score_gufunc",
    "_vrenergy_score_gufunc_w",
]
