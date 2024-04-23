from scoringrules.backend import backends

RUN_TESTS = ["numba", "jax", "torch"]
BACKENDS = [b for b in backends.available_backends if b in RUN_TESTS]

for backend in RUN_TESTS:
    backends.register_backend(backend)
