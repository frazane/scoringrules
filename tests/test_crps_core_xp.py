import numpy as np
import pytest
from scoringrules.backend import get_namespace
from scoringrules.core import crps

ESTIMATORS = ["nrg", "fair", "pwm", "qd", "int"]


@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_core_ensemble_matches_numpy(estimator, backend, to_backend):
    rng = np.random.default_rng(0)
    obs_np = rng.standard_normal(6)
    fct_np = np.sort(rng.standard_normal((6, 9)), axis=-1)
    obs, fct = to_backend(obs_np), to_backend(fct_np)
    out = np.asarray(crps.ensemble(obs, fct, estimator, xp=get_namespace(obs, fct)))
    ref = np.asarray(
        crps.ensemble(obs_np, fct_np, estimator, xp=get_namespace(obs_np, fct_np))
    )
    assert out == pytest.approx(ref, abs=1e-4)
    if estimator not in ("akr", "akr_circperm"):
        assert np.all(out >= -1e-6)


def test_core_ensemble_w_runs(backend, to_backend):
    rng = np.random.default_rng(1)
    obs_np = rng.standard_normal(5)
    fct_np = rng.standard_normal((5, 8))
    w_np = np.abs(rng.standard_normal((5, 8))) + 0.1
    w_np = w_np / w_np.sum(-1, keepdims=True)
    obs, fct, w = to_backend(obs_np), to_backend(fct_np), to_backend(w_np)
    out = np.asarray(crps.ensemble_w(obs, fct, w, "nrg", xp=get_namespace(obs, fct)))
    ref = np.asarray(
        crps.ensemble_w(obs_np, fct_np, w_np, "nrg", xp=get_namespace(obs_np, fct_np))
    )
    assert out == pytest.approx(ref, abs=1e-4)
