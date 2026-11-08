"""The input guard of ``use_numba``: the numba gufuncs run only for plain numpy
data, everything else takes the array path, also with ``backend="numba"`` and
under the global numba switch. Also parity cases that cut across the families
(M=1, bad ``nan_policy``)."""

import numpy as np
import pytest

import scoringrules as sr
from scoringrules._dispatch import gufunc_inputs_supported, use_numba
from scoringrules.core.utils import GufuncWrapper

from ._parity import global_numba, mv_v_func, mv_w_func, uv_w_func

rng = np.random.default_rng(0)
OBS, FCT = rng.normal(size=4), rng.normal(size=(4, 6))
ENS_W = rng.uniform(0.5, 2.0, size=(4, 6))
MOBS, MFCT = rng.normal(size=(4, 3)), rng.normal(size=(4, 8, 3))
MENS_W = rng.uniform(0.5, 2.0, size=(4, 8))
ALPHA = np.linspace(0.1, 0.9, 6)
SFCT = np.sort(FCT, axis=-1)


class Duck:
    """Implements ``__array__`` only, like a pandas Series or an xarray DataArray."""

    def __init__(self, a):
        self.a = a

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.a, dtype=dtype)


# --- gufunc_inputs_supported ---------------------------------------------------


def _read_only(a):
    a.setflags(write=False)
    return a


SUPPORTED = {
    "None": None,
    "float": 0.5,
    "int": 3,
    "bool": True,
    "f8": np.zeros(3),
    "f4": np.zeros(3, np.float32),
    "i1": np.zeros(3, np.int8),
    "u8": np.zeros(3, np.uint64),
    "bool array": np.zeros(3, bool),
    "big-endian": np.zeros(3, ">f8"),
    "fortran": np.zeros((3, 2), order="F"),
    "strided": np.zeros(6)[::2],
    "read-only": _read_only(np.zeros(3)),
    "0-d": np.asarray(0.5),
    "np.float64": np.float64(0.5),
    "np.float32": np.float32(0.5),
    "np.int64": np.int64(1),
    "list": [[0.5, 1.0], [2.0, 3.0]],
    "tuple": (0.5, 1.0),
    "int list": [1, 2],
}

UNSUPPORTED = {
    "masked": np.ma.array(np.zeros(3)),
    "matrix": np.asmatrix(np.zeros((2, 2))),
    "duck": Duck(np.zeros(3)),
    "f2": np.zeros(3, np.float16),
    "np.float16": np.float16(0.5),
    "f16 list": [np.float16(0.5)],
    "longdouble": np.zeros(3, np.longdouble),
    "np.longdouble": np.longdouble(0.5),
    "complex": np.zeros(3, complex),
    "Python complex": 0.5j,
    "object": np.zeros(3, object),
    "str list": ["a"],
    "datetime": np.zeros(3, "M8[s]"),
    "huge int": 2**70,
    "ragged list": [[1.0], [1.0, 2.0]],
    "dict": {"a": 1.0},
}


@pytest.mark.parametrize("name", list(SUPPORTED))
def test_gufunc_inputs_supported(name):
    assert gufunc_inputs_supported(np.zeros(3), SUPPORTED[name]) is True


@pytest.mark.parametrize("name", list(UNSUPPORTED))
def test_gufunc_inputs_unsupported(name):
    assert gufunc_inputs_supported(np.zeros(3), UNSUPPORTED[name]) is False


def test_use_numba_guard(monkeypatch, requires_numba):
    masked = np.ma.array(np.zeros(3))
    assert use_numba("numba", np.zeros(3), np.zeros((3, 2))) is True
    assert use_numba("numba", masked, np.zeros((3, 2))) is False
    assert use_numba("numba", np.zeros(3), np.zeros((3, 2)), aux=(None,)) is True
    assert use_numba("numba", np.zeros(3), np.zeros((3, 2)), aux=(masked,)) is False
    with global_numba():
        assert use_numba(None, np.zeros(3)) is True
        assert use_numba(None, np.zeros(3, np.float16)) is False
    # the switch is off again
    assert use_numba(None, np.zeros(3)) is False


def test_use_numba_framework_check_before_guard(requires_numba):
    # jax/torch obs or fct still raise for backend="numba"; as auxiliary inputs
    # they only send the call to the array path
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="numpy-compatible"):
        use_numba("numba", torch.zeros(3), np.zeros((3, 2)))
    assert (
        use_numba("numba", np.zeros(3), np.zeros((3, 2)), aux=(torch.ones(3),)) is False
    )
    with global_numba():
        assert use_numba(None, torch.zeros(3)) is False


# --- the public scores take the array path for unsupported inputs ---------------


@pytest.fixture()
def gufunc_calls(monkeypatch, requires_numba):
    """Count the gufunc calls (every gufunc goes through ``GufuncWrapper``)."""
    calls = []
    call = GufuncWrapper.__call__

    def counting(self, *args):
        calls.append(self.__name__)
        return call(self, *args)

    monkeypatch.setattr(GufuncWrapper, "__call__", counting)
    return calls


def _masked(a):
    return np.ma.array(a, mask=a > 1.0)


# one call per score family. The threshold-weighted scores are left out: they
# call the guard on the chained inputs, which their chains turn into plain
# ndarrays (float64 for the default chain of twcrps).
UV_SCORES = {
    "crps": (sr.crps_ensemble, (), {}),
    "crps_w": (sr.crps_ensemble, (), {"ens_w": ENS_W}),
    "owcrps": (sr.owcrps_ensemble, (), {"w_func": uv_w_func}),
    "gksuv": (sr.gksuv_ensemble, (), {}),
    "owgksuv": (sr.owgksuv_ensemble, (-0.5,), {}),
    "dssuv": (sr.dssuv_ensemble, (), {}),
    "ess": (sr.error_spread_score, (), {}),
    "quantile": (sr.crps_quantile, (ALPHA,), {}),
}
MV_SCORES = {
    "es": (sr.es_ensemble, (), {}),
    "es_w": (sr.es_ensemble, (), {"ens_w": MENS_W}),
    "owes": (sr.owes_ensemble, (mv_w_func,), {}),
    "vs": (sr.vs_ensemble, (), {}),
    "vrvs": (sr.vrvs_ensemble, (mv_w_func,), {}),
    "gksmv": (sr.gksmv_ensemble, (), {}),
    "dssmv": (sr.dssmv_ensemble, (), {}),
}


def _cases(obs, fct):
    """``{name: (obs, fct)}`` of inputs the gufuncs do not support."""
    return {
        "masked_obs": (_masked(obs), fct),
        "masked_fct": (obs, _masked(fct)),
        "float16": (obs.astype(np.float16), fct.astype(np.float16)),
        "longdouble": (obs.astype(np.longdouble), fct.astype(np.longdouble)),
        "complex_obs": (obs.astype(complex), fct),
        "duck": (Duck(obs), Duck(fct)),
    }


def _guard_params():
    for scores, obs, fct in ((UV_SCORES, OBS, SFCT), (MV_SCORES, MOBS, MFCT)):
        for score in scores:
            for case in _cases(obs, fct):
                yield pytest.param(scores[score], obs, fct, case, id=f"{score}-{case}")


def _outcome(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as err:  # noqa: BLE001
        return err


def _assert_same(res, ref):
    if isinstance(ref, Exception):
        assert type(res) is type(ref) and str(res) == str(ref), (res, ref)
        return
    assert type(res) is type(ref), (type(res), type(ref))
    assert np.asarray(res).dtype == np.asarray(ref).dtype
    np.testing.assert_array_equal(np.ma.getdata(res), np.ma.getdata(ref))
    np.testing.assert_array_equal(np.ma.getmaskarray(res), np.ma.getmaskarray(ref))


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::numpy.exceptions.ComplexWarning")
@pytest.mark.parametrize("spec,obs,fct,case", _guard_params())
def test_unsupported_inputs_take_the_array_path(spec, obs, fct, case, gufunc_calls):
    fn, args, kwargs = spec
    o, f = _cases(obs, fct)[case]
    ref = _outcome(fn, o, f, *args, **kwargs)
    _assert_same(_outcome(fn, o, f, *args, backend="numba", **kwargs), ref)
    with global_numba():
        _assert_same(_outcome(fn, o, f, *args, **kwargs), ref)
    assert gufunc_calls == []


@pytest.mark.parametrize(
    "fn,args,kwargs",
    [
        (sr.crps_ensemble, (OBS, FCT), {"ens_w": _masked(ENS_W)}),
        (sr.es_ensemble, (MOBS, MFCT), {"ens_w": _masked(MENS_W)}),
        (sr.vs_ensemble, (MOBS, MFCT, _masked(np.ones((3, 3)) * 1.5)), {}),
        (sr.vs_ensemble, (MOBS, MFCT), {"p": np.float16(1.5)}),
        (sr.crps_quantile, (OBS, SFCT, np.ma.array(ALPHA, mask=ALPHA > 0.5)), {}),
        (sr.crps_quantile, (OBS, SFCT, ALPHA.astype(np.longdouble)), {}),
        (
            sr.weighted_interval_score,
            (OBS, OBS, SFCT[:, :2], SFCT[:, 4:]),
            {"alpha": _masked(np.array([0.2, 0.5]) * 3), "w_median": 0.0},
        ),
    ],
    ids=[
        "crps-ens_w",
        "es-ens_w",
        "vs-w",
        "vs-p",
        "quantile-alpha",
        "quantile-alpha-f128",
        "wis-alpha",
    ],
)
def test_unsupported_auxiliary_inputs_take_the_array_path(
    fn, args, kwargs, gufunc_calls
):
    ref = _outcome(fn, *args, **kwargs)
    _assert_same(_outcome(fn, *args, backend="numba", **kwargs), ref)
    with global_numba():
        _assert_same(_outcome(fn, *args, **kwargs), ref)
    assert gufunc_calls == []


def test_explicit_numba_with_torch_ens_w_takes_the_array_path(gufunc_calls):
    torch = pytest.importorskip("torch")
    ens_w = torch.as_tensor(ENS_W)
    res = sr.crps_ensemble(OBS, FCT, ens_w=ens_w, backend="numba")
    np.testing.assert_allclose(res, sr.crps_ensemble(OBS, FCT, ens_w=ENS_W))
    assert gufunc_calls == []


def _supported_cases():
    return {
        "big_endian": (OBS.astype(">f8"), SFCT.astype(">f8")),
        "big_endian_f4": (OBS.astype(">f4"), SFCT.astype(">f4")),
        "fortran": (OBS, np.asfortranarray(SFCT)),
        "strided": (OBS[::2], SFCT[::2]),
        "uint8": (
            np.round(OBS * 2 + 3).astype(np.uint8),
            np.round(SFCT * 2 + 3).astype(np.uint8),
        ),
        "tuple": (tuple(OBS), tuple(map(tuple, SFCT))),
        "np_float32_obs": (np.float32(OBS[0]), SFCT[0].astype(np.float32)),
    }


_UNSIGNED_WRAPS = pytest.mark.xfail(
    strict=True,
    reason="the array path takes differences of unsigned integers in their own "
    "dtype, which wraps around; the gufuncs cast to float first (found in PR 5, "
    "not fixed, see notes)",
)


def _supported_params():
    for score in ["crps", "crps_w", "gksuv", "dssuv", "ess", "quantile"]:
        for case in _supported_cases():
            marks = (
                [_UNSIGNED_WRAPS]
                if case == "uint8" and score in ("crps_w", "gksuv")
                else []
            )
            yield pytest.param(score, case, marks=marks, id=f"{score}-{case}")


@pytest.mark.parametrize("score,case", _supported_params())
def test_supported_inputs_take_the_gufunc_path(score, case, gufunc_calls, paths_agree):
    fn, args, kwargs = UV_SCORES[score]
    obs, fct = _supported_cases()[case]
    if case in ("strided", "np_float32_obs") and "ens_w" in kwargs:
        kwargs = {"ens_w": ENS_W[::2] if case == "strided" else ENS_W[0]}
    tol = {}
    if score == "quantile" and np.asarray(fct).dtype.itemsize == 4:
        # float32 data with float64 alpha: float64 out, partly float32 inside
        tol = {"rtol": 1e-6}
    paths_agree(fn, obs, fct, *args, **tol, **kwargs)
    assert len(gufunc_calls) == 2  # backend="numba" and the global switch


@pytest.mark.parametrize(
    "fn,kwargs",
    [
        (sr.dssuv_ensemble, {"bias": False}),
        (sr.dssuv_ensemble, {"bias": True}),
        (sr.error_spread_score, {}),
    ],
    ids=["dssuv", "dssuv-bias", "ess"],
)
def test_bool_inputs_take_the_gufunc_path(fn, kwargs, gufunc_calls, paths_agree):
    # the array path accepts bool here (float64 out), so both paths must agree
    obs, fct = OBS > 0, np.concatenate([FCT > 0, FCT > -1], axis=-1)
    paths_agree(fn, obs, fct, allow_all_nan=True, **kwargs)
    assert len(gufunc_calls) == 2


# --- cross-family parity: M=1 and bad nan_policy -------------------------------

M1_UV = {
    **{
        f"crps-{e}": (sr.crps_ensemble, (), {"estimator": e})
        for e in ("nrg", "fair", "pwm", "qd", "akr", "akr_circperm", "int")
    },
    "owcrps": (sr.owcrps_ensemble, (-0.5, 1.0), {}),
    "vrcrps": (sr.vrcrps_ensemble, (-0.5, 1.0), {}),
    **{
        f"gksuv-{e}": (sr.gksuv_ensemble, (), {"estimator": e})
        for e in ("nrg", "fair", "akr", "akr_circperm")
    },
    "owgksuv": (sr.owgksuv_ensemble, (-0.5, 1.0), {}),
    "vrgksuv": (sr.vrgksuv_ensemble, (-0.5, 1.0), {}),
    "dssuv": (sr.dssuv_ensemble, (), {}),
}
M1_MV = {
    **{
        f"es-{e}": (sr.es_ensemble, (), {"estimator": e})
        for e in ("nrg", "fair", "akr", "akr_circperm")
    },
    "owes": (sr.owes_ensemble, (mv_w_func,), {}),
    "vres": (sr.vres_ensemble, (mv_w_func,), {}),
    **{f"vs-{e}": (sr.vs_ensemble, (), {"estimator": e}) for e in ("nrg", "fair")},
    "owvs": (sr.owvs_ensemble, (mv_w_func,), {}),
    "vrvs": (sr.vrvs_ensemble, (mv_w_func,), {}),
    **{
        f"gksmv-{e}": (sr.gksmv_ensemble, (), {"estimator": e})
        for e in ("nrg", "fair", "akr", "akr_circperm")
    },
    "owgksmv": (sr.owgksmv_ensemble, (mv_w_func,), {}),
    "vrgksmv": (sr.vrgksmv_ensemble, (mv_w_func,), {}),
    "dssmv": (sr.dssmv_ensemble, (), {}),
}


def _m1_params():
    for scores, mv in ((M1_UV, False), (M1_MV, True)):
        for name in scores:
            for weighted in (False, True):
                if weighted and name.startswith("dss"):
                    continue
                yield pytest.param(
                    scores[name], mv, weighted, id=f"{name}-{'w' if weighted else 'uw'}"
                )


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("spec,mv,weighted", _m1_params())
def test_single_member_paths_agree(spec, mv, weighted, paths_agree):
    """M=1: the fair estimators and the DSS are undefined (NaN or inf), the
    others reduce to their single-member form."""
    fn, args, kwargs = spec
    r = np.random.default_rng(3)
    obs, fct = (MOBS, r.normal(size=(4, 1, 3))) if mv else (OBS, r.normal(size=(4, 1)))
    if weighted:
        kwargs = {**kwargs, "ens_w": r.uniform(0.5, 2.0, size=(4, 1))}
    paths_agree(fn, obs, fct, *args, allow_all_nan=True, **kwargs)


NAN_POLICY_SCORES = {
    "crps": (sr.crps_ensemble, OBS, FCT, ()),
    "twcrps": (sr.twcrps_ensemble, OBS, FCT, (-0.5, 1.0)),
    "owcrps": (sr.owcrps_ensemble, OBS, FCT, (-0.5, 1.0)),
    "vrcrps": (sr.vrcrps_ensemble, OBS, FCT, (-0.5, 1.0)),
    "es": (sr.es_ensemble, MOBS, MFCT, ()),
    "twes": (sr.twes_ensemble, MOBS, MFCT, (mv_v_func,)),
    "owes": (sr.owes_ensemble, MOBS, MFCT, (mv_w_func,)),
    "vres": (sr.vres_ensemble, MOBS, MFCT, (mv_w_func,)),
}


@pytest.mark.parametrize("weighted", [False, True], ids=["uw", "w"])
@pytest.mark.parametrize("name", list(NAN_POLICY_SCORES))
def test_bad_nan_policy_raises(name, weighted, paths_agree):
    fn, obs, fct, args = NAN_POLICY_SCORES[name]
    kwargs = {"ens_w": ENS_W if obs.ndim == 1 else MENS_W} if weighted else {}
    paths_agree(fn, obs, fct, *args, nan_policy="bogus", raises=ValueError, **kwargs)
