import typing as tp

if tp.TYPE_CHECKING:
    from jax import Array as JaxArray
    from numpy.typing import NDArray
    from torch import Tensor as torchTensor

    _array = NDArray | JaxArray | torchTensor
    Array = tp.TypeVar("Array", bound=_array)
    ArrayLike = tp.TypeVar("ArrayLike", bound=_array | float | int)

    # "numpy", "jax" and "torch" are deprecated (removed in 1.0): only None | "numba"
    Backend = tp.Literal["numpy", "numba", "jax", "torch"] | None
    NanPolicy = tp.Literal["propagate", "omit", "raise"]
