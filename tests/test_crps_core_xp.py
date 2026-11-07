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


def _closed_ref(fn, *args):
    return np.asarray(
        fn(
            *[np.asarray(a) for a in args],
            xp=get_namespace(*[np.asarray(a) for a in args]),
        )
    )


@pytest.mark.parametrize(
    "dist,args",
    [
        ("normal", (0.3, 0.0, 1.0)),
        ("logistic", (0.3, 0.0, 1.0)),
        ("exponential", (0.8, 3.0)),
        ("gamma", (0.5, 2.0, 1.5)),
        ("poisson", (2.0, 3.0)),
    ],
)
def test_core_closed_matches_numpy(dist, args, backend, to_backend):
    from scoringrules.core import crps

    fn = getattr(crps, dist)
    bargs = [to_backend(np.asarray(float(a))) for a in args]
    out = np.asarray(fn(*bargs, xp=get_namespace(*bargs)))
    ref = _closed_ref(fn, *args)
    assert out == pytest.approx(ref, abs=1e-4)


def test_core_beta_works_numpy_jax_blocks_torch(backend, to_backend):
    from scoringrules.core import crps

    args = (0.3, 0.7, 1.1)
    bargs = [to_backend(np.asarray(float(a))) for a in args]
    if backend == "torch":
        with pytest.raises(NotImplementedError):
            crps.beta(*bargs, xp=get_namespace(*bargs))
    else:
        out = np.asarray(crps.beta(*bargs, xp=get_namespace(*bargs)))
        ref = _closed_ref(crps.beta, *args)
        assert out == pytest.approx(ref, abs=1e-4)
