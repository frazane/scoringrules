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


@pytest.mark.skip(
    reason="special-function methods exercised once extensions.py lands (later task)"
)
def test_special_function_method_present():
    pass


def test_mixed_namespaces_raise():
    torch = pytest.importorskip("torch")
    a = np.asarray([1.0, 2.0])
    b = torch.tensor([1.0, 2.0])
    with pytest.raises(ValueError, match="single framework|multiple|mixed"):
        get_namespace(a, b)
