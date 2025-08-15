from ._approx import (
    ensemble_uv,
    ow_ensemble_uv,
    vr_ensemble_uv,
    ensemble_mv,
    ow_ensemble_mv,
    vr_ensemble_mv,
)

from ._approx_w import (
    ensemble_uv_w,
    ow_ensemble_uv_w,
    vr_ensemble_uv_w,
    ensemble_mv_w,
    ow_ensemble_mv_w,
    vr_ensemble_mv_w,
)

try:
    from ._gufuncs import estimator_gufuncs_uv, estimator_gufuncs_mv
    from ._gufuncs_w import estimator_gufuncs_uv_w, estimator_gufuncs_mv_w
except ImportError:
    estimator_gufuncs_uv = None
    estimator_gufuncs_mv = None
    estimator_gufuncs_uv_w = None
    estimator_gufuncs_mv_w = None

__all__ = [
    "ensemble_uv",
    "ow_ensemble_uv",
    "vr_ensemble_uv",
    "ensemble_mv",
    "ow_ensemble_mv",
    "vr_ensemble_mv",
    "ensemble_uv_w",
    "ow_ensemble_uv_w",
    "vr_ensemble_uv_w",
    "ensemble_mv_w",
    "ow_ensemble_mv_w",
    "vr_ensemble_mv_w",
    "estimator_gufuncs_uv",
    "estimator_gufuncs_mv",
    "estimator_gufuncs_uv_w",
    "estimator_gufuncs_mv_w",
]
