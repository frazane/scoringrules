import os
from pathlib import Path

import numpy as np
import pytest
from scoringrules.backend import backends

DATA_DIR = Path(__file__).parent / "data"

if os.getenv("SR_TEST_OUTPUT", "False").lower() in ("true", "1", "t"):
    OUT_DIR = Path(__file__).parent / "output"
    OUT_DIR.mkdir(exist_ok=True)
else:
    OUT_DIR = None


def pytest_addoption(parser):
    """Add custom command-line options for pytest."""
    parser.addoption(
        "--backend",
        action="store",
        default=None,
        help="Specify backend to test",
    )


def get_test_backends(config):
    """Determine which backends to test."""
    backend_option = config.getoption("--backend")

    if backend_option:
        requested = backend_option.split(",")
    else:
        requested = ["numpy", "numba", "jax", "torch"]

    available = backends.available_backends
    test_backends = [b for b in requested if b in available]

    # Register backends
    for b in test_backends:
        try:
            backends.register_backend(b)
        except Exception as e:
            print(f"Warning: Could not register backend '{b}': {e}")

    return test_backends


# This generates the parametrization
def pytest_generate_tests(metafunc):
    if "backend" in metafunc.fixturenames:
        backends_to_test = get_test_backends(metafunc.config)
        if not backends_to_test:
            pytest.fail("No backends available for testing")
        metafunc.parametrize("backend", backends_to_test)


@pytest.fixture()
def probability_forecasts():
    data = np.genfromtxt(
        DATA_DIR / "precip_Niamey_2016.csv",
        delimiter=",",
        skip_header=1,
        usecols=(1, -1),
    )
    return data


@pytest.fixture()
def to_backend(backend):
    """Return a function that converts numpy input to the backend's native array.

    numpy/numba -> numpy ndarray; jax -> jax array; torch -> torch tensor.
    """
    import numpy as np

    if backend in ("numpy", "numba"):
        return lambda x: np.asarray(x)
    if backend == "jax":
        import jax.numpy as jnp

        return lambda x: jnp.asarray(x)
    if backend == "torch":
        import torch

        return lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float64)
    raise ValueError(backend)


@pytest.fixture()
def backend_kwargs(backend):
    """kwargs to pass to a score: only numba needs an explicit backend string."""
    return {"backend": "numba"} if backend == "numba" else {}


def assert_inferred(result, backend):
    """Assert the result array belongs to the expected framework (guards against
    silent numpy fallback when a non-numpy backend was requested)."""
    mod = type(result).__module__
    if backend == "jax":
        assert "jax" in mod, f"expected a jax array, got {mod}"
    elif backend == "torch":
        assert "torch" in mod, f"expected a torch tensor, got {mod}"
    else:
        assert "numpy" in mod, f"expected a numpy array, got {mod}"
