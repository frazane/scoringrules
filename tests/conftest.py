import os
from pathlib import Path

import numpy as np
import pytest
from scoringrules.backend import backends

DATA_DIR = Path(__file__).parent / "data"
RUN_TESTS = ["numpy", "numba", "jax", "torch"]
BACKENDS = [b for b in backends.available_backends if b in RUN_TESTS]

if os.getenv("SR_TEST_OUTPUT", "False").lower() in ("true", "1", "t"):
    OUT_DIR = Path(__file__).parent / "output"
    OUT_DIR.mkdir(exist_ok=True)
else:
    OUT_DIR = None

for backend in RUN_TESTS:
    backends.register_backend(backend)


@pytest.fixture()
def probability_forecasts():
    data = np.genfromtxt(
        DATA_DIR / "precip_Niamey_2016.csv",
        delimiter=",",
        skip_header=1,
        usecols=(1, -1),
    )

    return data
