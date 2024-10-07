from ._approx import (
    ensemble_uv,
    ow_ensemble_uv,
    vr_ensemble_uv,
    ensemble_mv,
    ow_ensemble_mv,
    vr_ensemble_mv,
)

try:
    from ._gufuncs import estimator_gufuncs
except ImportError:
    estimator_gufuncs = None

try:
    from ._gufuncs import estimator_gufuncs_mv
except ImportError:
    estimator_gufuncs_mv = None

__all__ = [
    "ensemble_uv",
    "ow_ensemble_uv",
    "vr_ensemble_uv",
    "ensemble_mv",
    "ow_ensemble_mv",
    "vr_ensemble_mv",
    "estimator_gufuncs",
    "estimator_gufuncs_mv",
]
