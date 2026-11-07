"""Shared test helpers: gufunc/array-path parity and float32 inputs.

The numba gufuncs and the array-API path must give the same scores. Use the
``paths_agree`` fixture (``tests/conftest.py``), which skips when numba is not
selected, rather than importing ``assert_paths_agree`` directly.
"""

import numpy as np


def assert_paths_agree(fn, *args, rtol=1e-10, atol=1e-12, **kwargs):
    """Assert ``fn(*args, backend="numba", **kwargs)`` matches the inferred
    numpy path ``fn(*args, **kwargs)`` and that neither modifies its inputs.

    Both results go through ``np.asarray`` (lazy gufuncs return Python scalars
    for 0-d ``obs``). Returns the two results as numpy arrays.
    """
    inputs = [a for a in (*args, *kwargs.values()) if isinstance(a, np.ndarray)]
    snapshot = [a.copy() for a in inputs]
    res_numba = np.asarray(fn(*args, backend="numba", **kwargs))
    res_numpy = np.asarray(fn(*args, **kwargs))
    for before, after in zip(snapshot, inputs, strict=True):
        np.testing.assert_array_equal(after, before, "input modified in place")
    assert res_numba.shape == res_numpy.shape, (res_numba.shape, res_numpy.shape)
    assert res_numba.dtype.kind == "f", f"numba path returned {res_numba.dtype}"
    assert res_numpy.dtype.kind == "f", f"numpy path returned {res_numpy.dtype}"
    np.testing.assert_allclose(res_numba, res_numpy, rtol=rtol, atol=atol)
    return res_numba, res_numpy


def as_float32(x, backend):
    """``x`` as a float32 array of ``backend``'s framework (numba -> numpy)."""
    x = np.asarray(x, dtype=np.float32)
    if backend in ("numpy", "numba"):
        return x
    if backend == "jax":
        import jax.numpy as jnp

        return jnp.asarray(x, dtype=jnp.float32)
    if backend == "torch":
        import torch

        return torch.as_tensor(x, dtype=torch.float32)
    raise ValueError(backend)
