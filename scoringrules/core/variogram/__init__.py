try:
    from ._gufuncs import estimator_gufuncs
    from ._gufuncs_w import estimator_gufuncs_w
except ImportError:
    estimator_gufuncs = None
    estimator_gufuncs_w = None

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
