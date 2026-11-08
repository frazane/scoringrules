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

from scoringrules.core.utils import lazy_gufunc_getattr

# numba gufuncs, imported on first access (``import scoringrules`` must not import numba)
__getattr__ = lazy_gufunc_getattr(
    __name__,
    {
        "estimator_gufuncs_uv": "._gufuncs",
        "estimator_gufuncs_mv": "._gufuncs",
        "estimator_gufuncs_uv_w": "._gufuncs_w",
        "estimator_gufuncs_mv_w": "._gufuncs_w",
    },
)


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
