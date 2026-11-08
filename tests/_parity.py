"""Shared test helpers: gufunc/array-path parity and float32 inputs.

The numba gufuncs and the array-API path must give the same scores. Use the
``paths_agree`` fixture (``tests/conftest.py``), which skips when numba is not
selected, rather than importing ``assert_paths_agree`` directly.
"""

import contextlib
import inspect

import numpy as np
import pytest

import scoringrules as sr
from scoringrules._dispatch import AUTO_EXCLUDED
from scoringrules.core.utils import GufuncWrapper

# default (rtol, atol) of ``assert_paths_agree`` by result dtype
_TOLERANCES = {np.dtype(np.float64): (1e-10, 1e-12), np.dtype(np.float32): (1e-5, 1e-6)}


# the score that a threshold-weighted score chains to, whose key it uses
_CHAINED_TO = {
    "twcrps_ensemble": "crps_ensemble",
    "twgksuv_ensemble": "gksuv_ensemble",
    "twgksmv_ensemble": "gksmv_ensemble",
    "twes_ensemble": "es_ensemble",
    "twvs_ensemble": "vs_ensemble",
}
# the scores whose ``use_numba`` key is ``(name, estimator)``
_ESTIMATOR_KEYED = {
    "crps_ensemble",
    "gksuv_ensemble",
    "gksmv_ensemble",
    "es_ensemble",
    "vs_ensemble",
}


def auto_uses_the_gufunc(fn, args, kwargs):
    """Whether ``numba="auto"`` should run the gufunc of ``fn(*args, **kwargs)``
    (for inputs the guard accepts): not if its key is in
    ``_dispatch.AUTO_EXCLUDED`` for the float dtype numpy promotes ``obs`` and
    ``fct`` to (float64 for integers). Recomputed here rather than asked from
    ``use_numba``, so that the dispatch is checked against the table."""
    name = getattr(fn, "__name__", "")
    name = _CHAINED_TO.get(name, name)
    key = name
    if name in _ESTIMATOR_KEYED:
        default = inspect.signature(fn).parameters["estimator"].default
        key = (name, kwargs.get("estimator", default))
    if key not in AUTO_EXCLUDED:
        return True
    dtypes = AUTO_EXCLUDED[key]
    if dtypes is None:
        return False
    dtype = np.result_type(*(np.asarray(a) for a in args[:2]))
    if dtype.kind != "f":
        dtype = np.dtype(np.float64)
    return dtype.newbyteorder("=") not in dtypes


def assert_paths_agree(
    fn,
    *args,
    rtol=None,
    atol=None,
    check_dtype=True,
    allow_all_nan=False,
    raises=None,
    expect_gufunc=True,
    auto_gufunc=None,
    **kwargs,
):
    """Assert that ``fn(*args, **kwargs)`` under ``config_context(numba=True)``
    (the gufunc) matches it under ``config_context(numba=False)`` (the array
    path), that the default ``numba="auto"`` gives exactly the result of the
    path it should take, and that none of them modifies its inputs.

    The results must have the same Python type (numpy scalars for 0-d results,
    plain ndarrays otherwise) and, through ``np.asarray``, the same shape, the
    same dtype (unless ``check_dtype=False``, for families where only shapes
    and values matter) and agree within ``rtol``/``atol``, NaN positions
    included. The default tolerances depend on the result dtype (1e-10/1e-12
    for float64, 1e-5/1e-6 for float32). An all-NaN result fails unless
    ``allow_all_nan=True``, so that two paths that both break cannot agree by
    accident. With ``raises``, all three must raise that exception instead.

    The ``numba=True`` call must call a numba gufunc and the ``numba=False``
    call none, so a guard regression cannot turn a parity test into a
    comparison of the array path with itself. Pass ``expect_gufunc=False`` for
    inputs that ``use_numba`` sends to the array path (then no call may run a
    gufunc). With ``expect_gufunc``, ``numba="auto"`` must call a gufunc iff
    ``auto_gufunc`` (by default ``auto_uses_the_gufunc``, from the exclusion
    table). Returns the numba and numpy results as numpy arrays.
    """
    inputs = [a for a in (*args, *kwargs.values()) if isinstance(a, np.ndarray)]
    snapshot = [a.copy() for a in inputs]

    def check_unchanged(path):
        for before, after in zip(snapshot, inputs, strict=True):
            np.testing.assert_array_equal(
                after, before, f"{path} path modified an input in place"
            )

    def run(numba):
        with sr.config_context(numba=numba), count_gufunc_calls() as calls:
            if raises is None:
                return fn(*args, **kwargs), calls
            with pytest.raises(raises):
                fn(*args, **kwargs)
            return None, calls

    raw_numba, numba_calls = run(True)
    check_unchanged("numba")
    raw_auto, auto_calls = run("auto")
    check_unchanged("auto")
    raw_numpy, numpy_calls = run(False)
    check_unchanged("numpy")
    if raises is not None:
        return None
    assert not numpy_calls, f"numba=False called {numpy_calls}"
    if expect_gufunc:
        assert numba_calls, "numba=True did not call a gufunc"
        if auto_gufunc is None:
            auto_gufunc = auto_uses_the_gufunc(fn, args, kwargs)
        if auto_gufunc:
            assert auto_calls, "numba='auto' did not call a gufunc"
        else:
            assert not auto_calls, f"numba='auto' called {auto_calls} (excluded)"
    else:
        assert not numba_calls and not auto_calls, (numba_calls, auto_calls)
    # the result of the path auto mode took, bit for bit
    raw_ref, ref_name = (raw_numba, "True") if auto_calls else (raw_numpy, "False")
    assert type(raw_auto) is type(raw_ref), (type(raw_auto), type(raw_ref))
    res_auto, res_ref = np.asarray(raw_auto), np.asarray(raw_ref)
    assert res_auto.dtype == res_ref.dtype, (res_auto.dtype, res_ref.dtype)
    np.testing.assert_array_equal(
        res_auto, res_ref, f"numba='auto' differs from numba={ref_name}"
    )
    assert type(raw_numba) is type(raw_numpy), (type(raw_numba), type(raw_numpy))
    res_numba, res_numpy = np.asarray(raw_numba), np.asarray(raw_numpy)
    assert res_numba.shape == res_numpy.shape, (res_numba.shape, res_numpy.shape)
    assert res_numba.dtype.kind == "f", f"numba path returned {res_numba.dtype}"
    assert res_numpy.dtype.kind == "f", f"numpy path returned {res_numpy.dtype}"
    if check_dtype:
        assert res_numba.dtype == res_numpy.dtype, (res_numba.dtype, res_numpy.dtype)
    if not allow_all_nan:
        assert not np.all(np.isnan(res_numpy)), "all-NaN result"
    default_rtol, default_atol = _TOLERANCES.get(
        res_numpy.dtype, _TOLERANCES[np.dtype(np.float32)]
    )
    np.testing.assert_allclose(
        res_numba,
        res_numpy,
        rtol=default_rtol if rtol is None else rtol,
        atol=default_atol if atol is None else atol,
    )
    return res_numba, res_numpy


@contextlib.contextmanager
def count_gufunc_calls():
    """Yield a list that gets one entry (the gufunc name) per numba gufunc call
    made inside the block; every gufunc goes through ``GufuncWrapper``."""
    calls = []
    original = GufuncWrapper.__call__

    def counting(self, *args):
        calls.append(self.__name__)
        return original(self, *args)

    GufuncWrapper.__call__ = counting
    try:
        yield calls
    finally:
        GufuncWrapper.__call__ = original


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


UV_CASES = (
    "default",
    "m_axis",
    "n1",
    "int_obs",
    "scalar_obs",
    "batch",
    "list_inputs",
    "int_fct",
    "nan_propagate",
    "nan_omit",
    "nan_raise",
    "nan_omit_m_axis",
    "float32",
    "zero_d_obs",
    "np_scalar_obs",
)


def uv_ensemble_inputs(case, M, weighted, seed=0, N=5):
    """``(obs, fct, kwargs)`` for a univariate ensemble parity case.

    ``case`` is one of ``UV_CASES``:

    - ``"default"``: ``fct`` of shape ``(N, M)``;
    - ``"m_axis"``: ``fct`` of shape ``(M, N)`` with ``m_axis=0``;
    - ``"n1"``: ``N=1``;
    - ``"int_obs"`` / ``"int_fct"``: integer ``obs`` / integer ``fct`` (with ties);
    - ``"scalar_obs"``: a Python float ``obs`` with ``fct`` of shape ``(M,)``;
      ``"zero_d_obs"`` and ``"np_scalar_obs"`` the same with a 0-d ndarray and
      a ``np.float64`` ``obs``;
    - ``"float32"``: ``obs``, ``fct`` (and ``ens_w``) as float32;
    - ``"batch"``: an extra leading axis, ``fct`` of shape ``(2, N, M)``;
    - ``"list_inputs"``: ``obs``, ``fct`` (and ``ens_w``) as nested lists;
    - ``"nan_<policy>"``: NaN members in row 0, a NaN ``obs`` in row 4 and (when
      weighted) a NaN weight in row 2, with ``nan_policy=<policy>``;
      ``"nan_omit_m_axis"`` is ``"nan_omit"`` laid out as ``"m_axis"``.

    With ``weighted``, ``kwargs`` holds non-uniform ``ens_w`` laid out like ``fct``.
    """
    rng = np.random.default_rng(seed)
    batch = {"n1": (1,), "batch": (2, N)}.get(case, (N,))
    obs = rng.normal(size=batch)
    fct = rng.normal(size=(*batch, M))
    w = rng.uniform(0.1, 2.0, size=(*batch, M))
    kwargs = {}
    if case.startswith("nan_"):
        fct[0, [1, 3]] = np.nan
        obs[4] = np.nan
        w[2, 0] = np.nan
        kwargs["nan_policy"] = case.split("_")[1]
    if case in ("m_axis", "nan_omit_m_axis"):
        fct, w = fct.T.copy(), w.T.copy()
        kwargs["m_axis"] = 0
    elif case == "int_obs":
        obs = rng.integers(-2, 3, size=batch)
    elif case == "int_fct":
        fct = rng.integers(-3, 4, size=fct.shape)
    elif case == "scalar_obs":
        obs, fct, w = float(obs[0]), fct[0], w[0]
    elif case == "zero_d_obs":
        obs, fct, w = np.asarray(obs[0]), fct[0], w[0]
    elif case == "np_scalar_obs":
        obs, fct, w = np.float64(obs[0]), fct[0], w[0]
    elif case == "float32":
        obs, fct, w = (x.astype(np.float32) for x in (obs, fct, w))
    elif case == "list_inputs":
        obs, fct, w = obs.tolist(), fct.tolist(), w.tolist()
    if weighted:
        kwargs["ens_w"] = w
    return obs, fct, kwargs


MV_CASES = (
    "default",
    "m_axis",
    "v_axis",
    "n1",
    "int_obs",
    "int_fct",
    "single",
    "batch",
    "list_inputs",
    "float32",
)

# NaN cases for the multivariate scores, kept out of ``MV_CASES`` because only
# the energy scores take ``nan_policy``; the others drop it and check that NaN
# propagates.
MV_NAN_CASES = ("nan_propagate", "nan_omit", "nan_raise")


def mv_ensemble_inputs(case, M, weighted, seed=0, N=5, D=3):
    """``(obs, fct, kwargs)`` for a multivariate ensemble parity case.

    ``case`` is one of ``MV_CASES``: ``"default"`` (``obs`` of shape ``(N, D)``,
    ``fct`` of shape ``(N, M, D)``), ``"m_axis"`` (``fct`` of shape ``(M, N, D)``
    with ``m_axis=0``), ``"v_axis"`` (``fct`` of shape ``(N, D, M)`` with
    ``m_axis=-1, v_axis=-2``), ``"n1"`` (``N=1``), ``"int_obs"`` (integer
    ``obs``), ``"int_fct"`` (integer ``fct``, with ties), ``"single"`` (``obs``
    of shape ``(D,)``, ``fct`` of shape ``(M, D)``), ``"batch"`` (an extra
    leading axis, ``fct`` of shape ``(2, N, M, D)``) and ``"list_inputs"``
    (``obs``, ``fct`` and ``ens_w`` as nested lists) and ``"float32"`` (float32
    ``obs``, ``fct`` and ``ens_w``). ``MV_NAN_CASES`` ``"nan_<policy>"`` put a
    NaN member element in sample 0, a NaN ``obs`` element in sample 4 and (when
    weighted) a NaN weight in sample 2, with ``nan_policy=<policy>``. With
    ``weighted``, ``kwargs`` holds non-uniform ``ens_w`` laid out like ``fct``
    without its variables axis.
    """
    rng = np.random.default_rng(seed)
    batch = {"n1": (1,), "single": (), "batch": (2, N)}.get(case, (N,))
    obs = rng.normal(size=(*batch, D))
    fct = rng.normal(size=(*batch, M, D))
    w = rng.uniform(0.1, 2.0, size=(*batch, M))
    kwargs = {}
    if case.startswith("nan_"):
        fct[0, 1, 0] = np.nan
        obs[4, -1] = np.nan
        w[2, 0] = np.nan
        kwargs["nan_policy"] = case.split("_")[1]
    if case == "m_axis":
        fct, w = np.moveaxis(fct, -2, 0).copy(), np.moveaxis(w, -1, 0).copy()
        kwargs["m_axis"] = 0
    elif case == "v_axis":
        fct = np.swapaxes(fct, -1, -2).copy()
        kwargs["m_axis"], kwargs["v_axis"] = -1, -2
    elif case == "int_obs":
        obs = rng.integers(-2, 3, size=obs.shape)
    elif case == "int_fct":
        fct = rng.integers(-3, 4, size=fct.shape)
    elif case == "list_inputs":
        obs, fct, w = obs.tolist(), fct.tolist(), w.tolist()
    elif case == "float32":
        obs, fct, w = (x.astype(np.float32) for x in (obs, fct, w))
    if weighted:
        kwargs["ens_w"] = w
    return obs, fct, kwargs


def mv_w_func(x):
    """A smooth, positive multivariate weight function (numpy only)."""
    return 1.0 / (1.0 + np.exp(-np.mean(x)))


def mv_v_func(x):
    """A multivariate chaining function (numpy only)."""
    return np.maximum(x, -0.5)


def uv_w_func(x):
    """A smooth, positive univariate weight function (numpy only)."""
    return 1.0 / (1.0 + np.exp(-x))
