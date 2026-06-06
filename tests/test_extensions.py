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


def test_gamma_grad_jax():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from array_api_compat import array_namespace
    from scoringrules.backend import extensions as ext

    g = jax.grad(lambda v: ext.gamma(array_namespace(v), v).sum())(
        jnp.asarray([1.5, 2.5])
    )
    assert jnp.all(jnp.isfinite(g))
