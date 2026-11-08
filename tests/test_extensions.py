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


def test_apply_along_axis_torch_python_float_keeps_precision():
    """The loop fallback (func1d not vmappable, here ``.item()``) keeps a float64
    input's precision for Python-float results instead of torch's float32 default."""
    torch = pytest.importorskip("torch")
    x = np.random.default_rng(0).normal(size=(3, 4))
    t = torch.tensor(x, dtype=torch.float64)
    out = ext.apply_along_axis(
        array_namespace(t), lambda a: 1.0 / 3.0 + a.mean().item(), t, -1
    )
    assert out.dtype == torch.float64
    np.testing.assert_allclose(out.numpy(), 1.0 / 3.0 + x.mean(axis=-1), rtol=1e-15)


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


def test_apply_along_axis_torch_rejects_out_of_range_axis():
    torch = pytest.importorskip("torch")
    x = torch.ones(2, 3)
    for axis in (2, -3):
        with pytest.raises(np.exceptions.AxisError):
            ext.apply_along_axis(array_namespace(x), torch.sum, x, axis)


def test_apply_along_axis_torch_empty_axis():
    torch = pytest.importorskip("torch")
    x = torch.ones(2, 0)
    out = ext.apply_along_axis(array_namespace(x), torch.sum, x, 1)
    np.testing.assert_array_equal(
        out.numpy(), np.apply_along_axis(np.sum, 1, np.ones((2, 0)))
    )


def test_kind_rejects_unknown_namespace():
    import types

    fake = types.ModuleType("array_api_strict")
    with pytest.raises(NotImplementedError, match="not supported"):
        ext._kind(fake)
    with pytest.raises(NotImplementedError):
        ext.erf(fake, np.asarray([0.5]))


def test_kind_known_namespaces(backend):
    mods = {"numpy": "numpy", "numba": "numpy", "jax": "jax.numpy", "torch": "torch"}
    xp_mod = pytest.importorskip(mods[backend])
    expected = {"numba": "numpy"}.get(backend, backend)
    assert ext._kind(array_namespace(xp_mod.ones(2))) == expected


@pytest.mark.parametrize("q", [0.3, [0.25, 0.75]])
@pytest.mark.parametrize("axis", [None, 0, -1])
@pytest.mark.parametrize("keepdims", [False, True])
def test_quantile_matches_numpy(backend, q, axis, keepdims):
    mods = {"numpy": "numpy", "numba": "numpy", "jax": "jax.numpy", "torch": "torch"}
    xp_mod = pytest.importorskip(mods[backend])
    from scoringrules.backend import get_namespace

    ref = np.random.default_rng(0).normal(size=(3, 8))
    x = xp_mod.asarray(ref)
    out = get_namespace(x).quantile(x, q, axis=axis, keepdims=keepdims)
    assert get_namespace(out)._xp is get_namespace(x)._xp
    expected = np.quantile(ref, q, axis=axis, keepdims=keepdims)
    assert np.asarray(out).shape == expected.shape
    assert out.dtype == x.dtype
    atol = 1e-6 if backend == "jax" else 0  # jax runs in float32 in the tests
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-6, atol=atol)


def test_quantile_torch_keeps_grad_dtype_and_device():
    torch = pytest.importorskip("torch")
    x = torch.tensor([[1.0, 3.0, 2.0, 5.0]], dtype=torch.float32, requires_grad=True)
    out = ext.quantile(array_namespace(x), x, [0.25, 0.5], axis=-1)
    assert out.dtype == torch.float32
    out.sum().backward()
    assert x.grad is not None and torch.all(torch.isfinite(x.grad))
    m = torch.zeros(2, 4, device="meta")
    assert ext.quantile(array_namespace(m), m, 0.5, axis=-1).device.type == "meta"


def test_quantile_jax_grad_and_jit():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    def f(v):
        return ext.quantile(array_namespace(v), v, 0.75, axis=-1).sum()

    x = jnp.asarray([[1.0, 3.0, 2.0, 5.0]])
    g = jax.grad(f)(x)
    assert jnp.all(jnp.isfinite(g))
    np.testing.assert_allclose(jax.jit(f)(x), np.quantile(np.asarray(x), 0.75))


@pytest.mark.parametrize("q", [0.3, [0.25, 0.75]])
def test_quantile_keeps_float32(backend, q):
    from ._parity import as_float32
    from scoringrules.backend import get_namespace

    x = as_float32(np.random.default_rng(1).normal(size=(3, 8)), backend)
    assert get_namespace(x).quantile(x, q, axis=-1).dtype == x.dtype


def test_factorial_comb_keep_float32_numpy():
    x = np.asarray([5.0, 6.0], dtype=np.float32)
    assert ext.factorial(NP, x).dtype == np.float32
    assert ext.comb(NP, x, x - 2).dtype == np.float32
    assert ext.factorial(NP, np.asarray([5, 6])).dtype == np.float64


def _comb_inputs(framework, dtype):
    n = [5.0, 171.0, 35.0, 5.5, 5.0, 5.0, -1.0]
    k = [2.0, 50.0, 14.0, 2.5, 7.0, -1.0, 0.0]
    if dtype == "float32":  # comb(171, 50) ~ 1e43 does not fit float32
        n[1], k[1] = 60.0, 30.0
    if framework == "numpy":
        return NP, np.asarray(n, dtype=dtype), np.asarray(k, dtype=dtype)
    if framework == "jax":
        jnp = pytest.importorskip("jax.numpy")
        return array_namespace(jnp.zeros(1)), jnp.asarray(n), jnp.asarray(k)
    torch = pytest.importorskip("torch")
    dt = getattr(torch, dtype)
    t = torch.tensor
    return array_namespace(t(0.0)), t(n, dtype=dt), t(k, dtype=dt)


@pytest.mark.parametrize(
    "framework, dtype",
    [("numpy", "float64"), ("numpy", "float32"), ("torch", "float64")]
    + [("torch", "float32"), ("jax", "float32")],
)
def test_comb_matches_scipy(framework, dtype):
    # scipy.special.comb(exact=False): real arguments, 0 outside 0 <= k <= n, and no
    # overflow for large n (the old factorial-based comb gave inf/nan from 171).
    import scipy.special as sp

    xp, n, k = _comb_inputs(framework, dtype)
    out = ext.comb(xp, n, k)
    assert str(out.dtype).endswith(dtype)
    expected = sp.comb(np.asarray(n, dtype=float), np.asarray(k, dtype=float))
    rtol = 1e-12 if dtype == "float64" else 1e-5
    np.testing.assert_allclose(np.asarray(out, dtype=float), expected, rtol=rtol)


def test_comb_grad_torch():
    torch = pytest.importorskip("torch")
    n = torch.tensor([10.0, 20.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: ext.comb(torch, x, x - 3.0), (n,))


def _cholesky_inputs():
    """A stack (2, 3, 3, 3): positive definite matrices except [0, 1] (singular),
    [1, 0] (indefinite), [1, 2] (a NaN entry) and [0, 2] (an inf entry)."""
    rng = np.random.default_rng(3)
    a = rng.normal(size=(2, 3, 3, 5))
    x = a @ np.swapaxes(a, -1, -2) / 5
    v = rng.normal(size=3)
    x[0, 1] = np.outer(v, v)
    x[1, 0] = np.diag([1.0, -1.0, 2.0])
    x[1, 2, 0, 1] = x[1, 2, 1, 0] = np.nan
    x[0, 2, 2, 2] = np.inf
    bad = np.zeros((2, 3), dtype=bool)
    bad[0, 1] = bad[1, 0] = bad[1, 2] = bad[0, 2] = True
    return x, bad


def test_cholesky_nan_for_non_positive_definite(backend):
    mods = {"numpy": "numpy", "numba": "numpy", "jax": "jax.numpy", "torch": "torch"}
    xp_mod = pytest.importorskip(mods[backend])
    from scoringrules.backend import get_namespace

    ref, bad = _cholesky_inputs()
    if backend == "torch":
        x = xp_mod.asarray(ref, dtype=xp_mod.float64)
    else:
        x = xp_mod.asarray(ref)
    L = get_namespace(x).cholesky(x)  # does not raise
    assert get_namespace(L)._xp is get_namespace(x)._xp
    L = np.asarray(L)
    assert L.shape == ref.shape
    assert np.isnan(L[bad]).all()
    rtol = 1e-5 if backend == "jax" else 1e-12  # jax runs in float32 in the tests
    np.testing.assert_allclose(L[~bad], np.linalg.cholesky(ref[~bad]), rtol=rtol)
    # a single matrix
    np.testing.assert_allclose(
        np.asarray(get_namespace(x).cholesky(x[0, 0])), L[0, 0], rtol=rtol
    )
    assert np.isnan(np.asarray(get_namespace(x).cholesky(x[1, 0]))).all()


def test_cholesky_numpy_isolates_failures_in_a_large_stack():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(1000, 3, 6))
    x = a @ np.swapaxes(a, -1, -2)
    x[[3, 500, 999]] = np.ones((3, 3))  # singular
    L = ext.cholesky(NP, x)
    bad = np.zeros(1000, dtype=bool)
    bad[[3, 500, 999]] = True
    assert np.isnan(L[bad]).all()
    np.testing.assert_array_equal(L[~bad], np.linalg.cholesky(x[~bad]))
