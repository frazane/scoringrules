"""Torch device / dtype placement of arrays created inside scoringrules (#139).

No GPU is needed. Two complementary tricks stand in for CUDA:

- ``meta`` tensors: torch's device check rejects mixing a non-scalar ``meta``
  tensor with a CPU one, exactly as it does for CUDA. Any array created on the
  default (CPU) device instead of the input's device makes the score raise. Meta
  tensors have no data, so this only works for scores without data-dependent
  Python control flow (``if xp.any(...)``, ``.item()``).
- ``_RequireExplicitDevice``: a ``TorchFunctionMode`` that fails any torch
  factory call (``arange``/``zeros``/``ones``/...) made without an explicit
  ``device=``. It runs on real CPU data, so it also covers the data-dependent
  scores the meta trick cannot reach.
"""

import warnings

import numpy as np
import pytest

import scoringrules as sr

torch = pytest.importorskip("torch")
from torch.overrides import TorchFunctionMode  # noqa: E402

META = "meta"

_FACTORIES = {
    torch.arange,
    torch.zeros,
    torch.ones,
    torch.full,
    torch.empty,
    torch.linspace,
    torch.logspace,
    torch.eye,
}


class _RequireExplicitDevice(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in _FACTORIES and kwargs.get("device") is None:
            raise AssertionError(f"torch.{func.__name__} called without device=")
        return func(*args, **kwargs)


@pytest.fixture(autouse=True)
def _silence_deprecations():
    # silence the DeprecationWarnings of ``backend="torch"`` and ``sr.backends``
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        yield


def _meta(*shape):
    return torch.zeros(shape, device=META)


# --- CRPS family -------------------------------------------------------------


@pytest.mark.parametrize(
    "estimator", ["nrg", "fair", "pwm", "qd", "int", "akr", "akr_circperm"]
)
def test_crps_ensemble_meta_device(estimator):
    # the reproducer from #139, for every estimator
    out = sr.crps_ensemble(_meta(4), _meta(4, 10), estimator=estimator)
    assert out.device.type == META
    assert out.shape == (4,)


def test_twcrps_ensemble_meta_device():
    assert sr.twcrps_ensemble(_meta(4), _meta(4, 10)).device.type == META


def test_crps_mixnorm_default_weights_meta_device():
    out = sr.crps_mixnorm(_meta(4), _meta(4, 3), torch.ones(4, 3, device=META))
    assert out.device.type == META


@pytest.mark.parametrize(
    "call",
    [
        lambda t: sr.crps_ensemble(t["obs"], t["fct"], estimator="qd"),
        lambda t: sr.crps_ensemble(t["obs"], t["fct"], estimator="pwm"),
        lambda t: sr.crps_ensemble(t["obs"], t["fct"], estimator="int"),
        lambda t: sr.crps_mixnorm(t["obs"], t["fct"], torch.ones_like(t["fct"])),
        lambda t: sr.crps_hypergeometric(t["k"], t["m"], t["n"], t["N"]),
        lambda t: sr.crps_hypergeometric(t["k"], t["m2"], t["n2"], t["N2"]),
    ],
    ids=["qd", "pwm", "int", "mixnorm", "hypergeo-scalar-n", "hypergeo-array-n"],
)
def test_crps_creation_calls_pass_device(call):
    f64 = torch.float64
    t = {
        "obs": torch.tensor([0.1, 0.5, 2.0], dtype=f64),
        "fct": torch.randn(3, 10, dtype=f64),
        "k": torch.tensor([2.0, 3.0], dtype=f64),
        "m": torch.tensor(5.0, dtype=f64),
        "n": torch.tensor(4.0, dtype=f64),
        "N": torch.tensor(3.0, dtype=f64),
        "m2": torch.tensor([5.0, 6.0], dtype=f64),
        "n2": torch.tensor([4.0, 3.0], dtype=f64),
        "N2": torch.tensor([3.0, 2.0], dtype=f64),
    }
    with _RequireExplicitDevice():
        out = call(t)
    assert out.dtype == f64


@pytest.mark.parametrize("estimator", ["pwm", "qd", "int"])
def test_crps_ensemble_created_arrays_follow_input_dtype(estimator):
    rng = np.random.default_rng(0)
    obs, fct = rng.normal(size=5), rng.normal(size=(5, 7))
    ref = sr.crps_ensemble(obs, fct, estimator=estimator)

    # torch float64: the created cdf levels / weights used to be float32
    out = sr.crps_ensemble(torch.tensor(obs), torch.tensor(fct), estimator=estimator)
    assert out.dtype == torch.float64
    np.testing.assert_allclose(out.numpy(), ref, rtol=1e-12)

    # numpy float32 stays float32 (created arrays used to default to float64)
    out32 = sr.crps_ensemble(
        obs.astype(np.float32), fct.astype(np.float32), estimator=estimator
    )
    assert out32.dtype == np.float32


def test_crps_mixnorm_default_weights_follow_input_dtype():
    # default weights used to be float32 for float64 torch input (~1e-7 error)
    m = np.array([[0.1, -0.3, 1.0]])
    s = np.array([[0.4, 2.1, 0.7]])
    ref = sr.crps_mixnorm(np.zeros(1), m, s)
    out = sr.crps_mixnorm(
        torch.zeros(1, dtype=torch.float64), torch.tensor(m), torch.tensor(s)
    )
    assert out.dtype == torch.float64
    np.testing.assert_allclose(out.numpy(), ref, rtol=1e-12)


@pytest.mark.parametrize("estimator", ["pwm", "qd", "int"])
def test_crps_ensemble_created_arrays_keep_autograd(estimator):
    fct = torch.randn(4, 10, dtype=torch.float64, requires_grad=True)
    out = sr.crps_ensemble(
        torch.zeros(4, dtype=torch.float64), fct, estimator=estimator
    )
    out.sum().backward()
    assert fct.grad is not None and torch.isfinite(fct.grad).all()


# --- deprecated backend string and the sr.backends shim ---------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda: sr.vs_ensemble(_meta(4, 3), _meta(4, 10, 3), backend="torch"),
        lambda: sr.twvs_ensemble(
            _meta(4, 3), _meta(4, 10, 3), v_func=lambda x: x, backend="torch"
        ),
        lambda: sr.logs_ensemble(_meta(4), _meta(4, 10), backend="torch"),
        lambda: sr.logs_mixnorm(
            _meta(4), _meta(4, 3), torch.ones(4, 3, device=META), backend="torch"
        ),
    ],
    ids=["vs", "twvs", "logs_ensemble", "logs_mixnorm"],
)
def test_legacy_scores_meta_device(call):
    assert call().device.type == META


@pytest.mark.parametrize(
    "call",
    [
        lambda t: sr.vs_ensemble(t["obs"], t["fct"], backend="torch"),
        lambda t: sr.owvs_ensemble(
            t["obs"], t["fct"], w_func=lambda x: x.sum() * 0 + 1.0, backend="torch"
        ),
        lambda t: sr.logs_ensemble(t["obs1"], t["fct1"], backend="torch"),
        lambda t: sr.logs_mixnorm(
            t["obs1"], t["fct1"], torch.ones_like(t["fct1"]), backend="torch"
        ),
        lambda t: sr.rps_score(t["cat"], t["prob"], onehot=False, backend="torch"),
    ],
    ids=["vs", "owvs", "logs_ensemble", "logs_mixnorm", "rps-categorical"],
)
def test_legacy_creation_calls_pass_device(call):
    f64 = torch.float64
    t = {
        "obs": torch.randn(3, 4, dtype=f64),
        "fct": torch.randn(3, 10, 4, dtype=f64),
        "obs1": torch.tensor([0.1, 0.5, 2.0], dtype=f64),
        "fct1": torch.randn(3, 10, dtype=f64),
        "cat": torch.tensor([1.0, 2.0], dtype=f64),
        "prob": torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]], dtype=f64),
    }
    with _RequireExplicitDevice():
        out = call(t)
    assert out.dtype == f64


def test_backends_shim_creation():
    # ``sr.backends["torch"]`` is the augmented array-API namespace of torch
    B = sr.backends["torch"]
    x = _meta(3)
    assert B.device(x).type == META
    for arr in (
        B.arange(3, device=B.device(x)),
        B.arange(1, 4, device=B.device(x)),
        B.zeros((2,), device=B.device(x)),
        B.ones((2,), device=B.device(x)),
    ):
        assert arr.device.type == META
    assert B.arange(3).tolist() == [0, 1, 2]
