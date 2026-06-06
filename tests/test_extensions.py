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
