import numpy as np
import pytest
from scoringrules.backend.namespace import get_namespace


def test_numpy_inference_standard_ops():
    x = np.asarray([1.0, -2.0, 3.0])
    xp = get_namespace(x)
    out = xp.sum(xp.abs(x))
    assert np.asarray(out) == pytest.approx(6.0)


def test_all_scalar_inputs_fall_back_to_numpy():
    xp = get_namespace(0.3, 0.7, 1.1)
    out = xp.asarray([1.0, 2.0])
    assert isinstance(np.asarray(out), np.ndarray)
    assert np.asarray(xp.sum(out)) == pytest.approx(3.0)


def test_list_inputs_fall_back_to_numpy():
    xp = get_namespace([1.0, 2.0, 3.0])
    out = xp.asarray([1.0, 2.0, 3.0])
    assert np.asarray(xp.sum(out)) == pytest.approx(6.0)


def test_norm_delegates_to_linalg():
    x = np.asarray([[3.0, 4.0]])
    xp = get_namespace(x)
    assert np.asarray(xp.norm(x, axis=-1)) == pytest.approx([5.0])


def test_gather_delegates_to_take_along_axis():
    x = np.asarray([[10.0, 20.0, 30.0]])
    ind = np.asarray([[2, 0, 1]])
    xp = get_namespace(x)
    out = np.asarray(xp.gather(x, ind, axis=-1))
    assert out.ravel().tolist() == pytest.approx([30.0, 10.0, 20.0])


def test_elementwise_math_coerces_python_scalars():
    # array-API torch rejects xp.sqrt(2.0); the wrapper must coerce scalars so
    # core code that does xp.sqrt(2.0) / xp.log(2) works on every backend.
    x = np.asarray([1.0])  # numpy namespace; behaviour must be uniform
    xp = get_namespace(x)
    assert np.asarray(xp.sqrt(2.0)) == pytest.approx(np.sqrt(2.0))
    assert np.asarray(xp.log(2.0)) == pytest.approx(np.log(2.0))


def test_elementwise_math_coerces_scalars_torch():
    torch = pytest.importorskip("torch")
    xp = get_namespace(torch.empty(1))
    assert float(xp.sqrt(2.0)) == pytest.approx(float(np.sqrt(2.0)))
    assert float(xp.sqrt(xp.pi)) == pytest.approx(float(np.sqrt(np.pi)))


def test_missing_xp_raises_attributeerror_not_recursion():
    from scoringrules.backend.namespace import ArrayAPINamespace

    obj = ArrayAPINamespace.__new__(ArrayAPINamespace)  # _xp never set
    with pytest.raises(AttributeError):
        obj.sum  # noqa: B018


def test_mixed_namespaces_raise():
    torch = pytest.importorskip("torch")
    a = np.asarray([1.0, 2.0])
    b = torch.tensor([1.0, 2.0])
    with pytest.raises(ValueError, match="single framework|multiple|mixed"):
        get_namespace(a, b)
