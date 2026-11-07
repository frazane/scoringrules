import numpy as np
import pytest
from scoringrules.backend import extensions as ext
from array_api_compat import array_namespace

NP = array_namespace(np.empty(0))


def test_erf_numpy():
    x = np.asarray([0.0, 1.0])
    out = np.asarray(ext.erf(NP, x))
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(0.8427007929, abs=1e-6)


def test_gamma_numpy():
    x = np.asarray([1.0, 2.0, 3.0])
    out = np.asarray(ext.gamma(NP, x))
    assert out == pytest.approx([1.0, 1.0, 2.0])


def test_betainc_numpy():
    out = np.asarray(ext.betainc(NP, np.asarray(2.0), np.asarray(3.0), np.asarray(0.5)))
    assert out == pytest.approx(0.6875, abs=1e-6)


def test_comb_numpy():
    out = np.asarray(ext.comb(NP, np.asarray(5), np.asarray(2)))
    assert out == pytest.approx(10.0)


def test_factorial_numpy():
    out = np.asarray(ext.factorial(NP, np.asarray(5)))
    assert out == pytest.approx(120.0)


def test_gammalinc_numpy():
    # lower incomplete gamma (unregularised): gamma(2)*P(2, 1) = 1 - 2/e
    out = np.asarray(ext.gammalinc(NP, np.asarray(2.0), np.asarray(1.0)))
    assert out == pytest.approx(0.2642411177, abs=1e-6)


def test_gammauinc_numpy():
    # upper incomplete gamma (unregularised): gamma(1)*Q(1, 0.5) = exp(-0.5)
    out = np.asarray(ext.gammauinc(NP, np.asarray(1.0), np.asarray(0.5)))
    assert out == pytest.approx(0.6065306597, abs=1e-6)


def test_apply_along_axis_numpy_both_axes():
    x = np.arange(6.0).reshape(2, 3)
    # scalar-returning func1d; check both the last axis and a non-last axis
    last = np.asarray(ext.apply_along_axis(NP, np.sum, x, -1))
    assert last.tolist() == pytest.approx([3.0, 12.0])
    first = np.asarray(ext.apply_along_axis(NP, np.sum, x, 0))
    assert first.tolist() == pytest.approx([3.0, 5.0, 7.0])


def test_apply_along_axis_jax_non_last_axis():
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from array_api_compat import array_namespace

    x = jnp.arange(6.0).reshape(2, 3)
    # regression guard: the vmap fast path must handle axis=0, not just axis=-1
    out = np.asarray(ext.apply_along_axis(array_namespace(x), jnp.sum, x, 0))
    assert out.tolist() == pytest.approx([3.0, 5.0, 7.0])


def test_erf_matches_across_backends(backend):
    mods = {"numpy": "numpy", "numba": "numpy", "jax": "jax.numpy", "torch": "torch"}
    xp_mod = pytest.importorskip(mods[backend])
    import numpy as np

    ref = np.asarray([0.1, 0.5, 1.0, 2.0])
    x = xp_mod.tensor(ref) if backend == "torch" else xp_mod.asarray(ref)
    from array_api_compat import array_namespace
    from scoringrules.backend import extensions as ext

    out = np.asarray(ext.erf(array_namespace(x), x))
    assert out == pytest.approx(
        np.asarray(ext.erf(array_namespace(ref), ref)), abs=1e-5
    )


def test_torch_betainc_blocked():
    pytest.importorskip("torch")
    import torch
    from array_api_compat import array_namespace
    from scoringrules.backend import extensions as ext

    a = torch.tensor([2.0])
    with pytest.raises(NotImplementedError):
        ext.betainc(array_namespace(a), a, torch.tensor([3.0]), torch.tensor([0.5]))


def test_torch_hypergeometric_and_expi_blocked():
    pytest.importorskip("torch")
    import torch
    from array_api_compat import array_namespace
    from scoringrules.backend import extensions as ext

    t = torch.tensor([0.5])
    ns = array_namespace(t)
    with pytest.raises(NotImplementedError):
        ext.hypergeometric(ns, t, t, t, t)
    with pytest.raises(NotImplementedError):
        ext.expi(ns, t)


def test_gamma_grad_jax():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from array_api_compat import array_namespace
    from scoringrules.backend import extensions as ext

    g = jax.grad(lambda v: ext.gamma(array_namespace(v), v).sum())(
        jnp.asarray([1.5, 2.5])
    )
    assert jnp.all(jnp.isfinite(g))


# --- torch: no numpy round-trips (device + autograd preserved) ---------------

_TORCH_NATIVE_SPECIAL = {
    "erf": 1,
    "gamma": 1,
    "gammainc": 2,
    "gammalinc": 2,
    "gammauinc": 2,
    "beta": 2,
    "mbessel0": 1,
    "mbessel1": 1,
    "factorial": 1,
}


@pytest.mark.parametrize("name", sorted(_TORCH_NATIVE_SPECIAL))
def test_torch_special_functions_stay_native(name):
    torch = pytest.importorskip("torch")
    nargs = _TORCH_NATIVE_SPECIAL[name]
    fn = getattr(ext, name)

    meta = [torch.full((3,), 1.5, device="meta") for _ in range(nargs)]
    assert fn(array_namespace(meta[0]), *meta).device.type == "meta"

    # torch implements no derivative of the incomplete gamma w.r.t. its shape
    # (first) argument, so for those only ``x`` requires grad.
    no_shape_grad = name in ("gammainc", "gammalinc", "gammauinc")
    args = [
        torch.tensor(
            [1.5, 2.5],
            dtype=torch.float64,
            requires_grad=not (no_shape_grad and i == 0),
        )
        for i in range(nargs)
    ]
    out = fn(array_namespace(args[-1]), *args)
    assert out.requires_grad
    out.sum().backward()
    assert all(a.grad is not None for a in args if a.requires_grad)


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_apply_along_axis_torch_matches_numpy(axis):
    torch = pytest.importorskip("torch")
    x = np.arange(24.0).reshape(2, 3, 4)
    t = torch.tensor(x)
    ns = array_namespace(t)
    # scalar-returning func1d (vmap path)
    out = ext.apply_along_axis(ns, torch.sum, t, axis)
    assert isinstance(out, torch.Tensor)
    np.testing.assert_allclose(out.numpy(), np.apply_along_axis(np.sum, axis, x))
    # vector-returning func1d: its output dimension replaces ``axis``
    out = ext.apply_along_axis(ns, lambda a: a[:2] * 2, t, axis)
    np.testing.assert_allclose(
        out.numpy(), np.apply_along_axis(lambda a: a[:2] * 2, axis, x)
    )


def test_apply_along_axis_torch_keeps_grad_and_device():
    torch = pytest.importorskip("torch")

    def func1d(a):
        return (a**2).sum()

    x = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    out = ext.apply_along_axis(array_namespace(x), func1d, x, -1)
    assert out.requires_grad
    out.sum().backward()
    torch.testing.assert_close(x.grad, 2 * x.detach())

    m = torch.zeros(3, 4, device="meta")
    out = ext.apply_along_axis(array_namespace(m), func1d, m, 0)
    assert out.device.type == "meta"
    assert out.shape == (4,)


def test_apply_along_axis_torch_loop_fallback_keeps_grad():
    torch = pytest.importorskip("torch")

    def func1d(a):
        # data-dependent Python control flow: torch.vmap cannot trace this,
        # so the per-slice loop fallback is used.
        return a.sum() if a.sum().item() > 0 else -a.sum()

    x = torch.tensor([[1.0, 2.0], [-3.0, -4.0]], requires_grad=True)
    out = ext.apply_along_axis(array_namespace(x), func1d, x, -1)
    assert out.tolist() == [3.0, 7.0]
    out.sum().backward()
    assert x.grad.tolist() == [[1.0, 1.0], [-1.0, -1.0]]


def test_indices_torch_native_device_and_dtype():
    torch = pytest.importorskip("torch")
    x = torch.zeros(1, device="meta")
    ns = array_namespace(x)
    out = ext.indices(ns, (2, 3), device=x.device)
    assert isinstance(out, torch.Tensor)
    assert out.device.type == "meta"
    assert out.shape == (2, 2, 3)
    assert out.dtype == torch.int64

    cpu = ext.indices(ns, (2, 3))
    np.testing.assert_array_equal(cpu.numpy(), np.indices((2, 3)))
    assert ext.indices(ns, ()).shape == (0,)
    assert ext.indices(ns, (2,), dtype=torch.float64).dtype == torch.float64


def test_indices_numpy_and_jax():
    np.testing.assert_array_equal(ext.indices(NP, (2, 3)), np.indices((2, 3)))
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    ns = array_namespace(jnp.zeros(1))
    out = ext.indices(ns, (2, 3), device=jax.devices()[0])
    np.testing.assert_array_equal(np.asarray(out), np.indices((2, 3)))
