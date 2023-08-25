from scoringrules import register_backend
from scoringrules.backend.registry import BackendNotAvailable

try:
    register_backend("jax")
    JAX_IMPORTED = True
except BackendNotAvailable:
    JAX_IMPORTED = False
