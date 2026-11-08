from scoringrules.core.utils import lazy_gufunc_getattr

# numba gufuncs, imported on first access (``import scoringrules`` must not import numba)
__getattr__ = lazy_gufunc_getattr(
    __name__,
    {
        "estimator_gufuncs": "._gufuncs",
        "estimator_gufuncs_w": "._gufuncs_w",
    },
)

from ._score import es_ensemble as es
from ._score import owes_ensemble as owes
from ._score import vres_ensemble as vres

from ._score_w import es_ensemble_w as es_w
from ._score_w import owes_ensemble_w as owes_w
from ._score_w import vres_ensemble_w as vres_w

__all__ = [
    "es",
    "es_w",
    "owes",
    "owes_w",
    "vres",
    "vres_w",
    "estimator_gufuncs",
    "estimator_gufuncs_w",
]
