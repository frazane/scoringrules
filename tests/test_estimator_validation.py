"""Invalid estimators raise ValueError on every path (#164), including the keys
used internally by the numba gufunc tables ("ownrg", "ow", ...)."""

import numpy as np
import pytest

import scoringrules as sr

rng = np.random.default_rng(0)
UV_OBS, UV_FCT = rng.normal(size=3), rng.normal(size=(3, 5))
MV_OBS, MV_FCT = rng.normal(size=(3, 2)), rng.normal(size=(3, 5, 2))
ENS_W = rng.uniform(0.5, 2.0, size=(3, 5))

INTERNAL = ["ownrg", "vrnrg", "ow", "vr", "bogus"]
TW = {"v_func": lambda x: x}

SCORES = {
    "crps": (sr.crps_ensemble, UV_OBS, UV_FCT, {}),
    "twcrps": (sr.twcrps_ensemble, UV_OBS, UV_FCT, TW),
    "es": (sr.es_ensemble, MV_OBS, MV_FCT, {}),
    "twes": (sr.twes_ensemble, MV_OBS, MV_FCT, TW),
    "vs": (sr.vs_ensemble, MV_OBS, MV_FCT, {}),
    "twvs": (sr.twvs_ensemble, MV_OBS, MV_FCT, TW),
    "gksuv": (sr.gksuv_ensemble, UV_OBS, UV_FCT, {}),
    "twgksuv": (sr.twgksuv_ensemble, UV_OBS, UV_FCT, TW),
    "gksmv": (sr.gksmv_ensemble, MV_OBS, MV_FCT, {}),
    "twgksmv": (sr.twgksmv_ensemble, MV_OBS, MV_FCT, TW),
}


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("estimator", INTERNAL)
@pytest.mark.parametrize("name", list(SCORES))
def test_invalid_estimator_raises_value_error(name, estimator, weighted, paths_agree):
    fn, obs, fct, kwargs = SCORES[name]
    if weighted:
        kwargs = {**kwargs, "ens_w": ENS_W}
    paths_agree(fn, obs, fct, estimator=estimator, raises=ValueError, **kwargs)
