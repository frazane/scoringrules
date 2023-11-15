from scoringrules import register_backend
from scoringrules.backend.registry import BackendNotAvailable

try:
    register_backend("jax")
    JAX_IMPORTED = True
except BackendNotAvailable:
    JAX_IMPORTED = False

try:
    register_backend("torch")
    TORCH_IMPORTED = True
except BackendNotAvailable:
    TORCH_IMPORTED = False

try:
    register_backend("tensorflow")
    TENSORFLOW_IMPORTED = True
except BackendNotAvailable:
    TENSORFLOW_IMPORTED = False
