from scoringrules.core.utils import lazy_gufunc_getattr

# numba gufuncs, imported on first access (``import scoringrules`` must not import numba)
__getattr__ = lazy_gufunc_getattr(
    __name__,
    {
        "estimator_gufuncs": "._gufuncs",
        "estimator_gufuncs_w": "._gufuncs_w",
    },
)

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
    "estimator_gufuncs",
    "estimator_gufuncs_w",
]
