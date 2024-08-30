import typing as tp

if tp.TYPE_CHECKING:
    from jax import Array as JaxArray
    from numpy.typing import NDArray
    from tensorflow import Tensor as tensorflowTensor
    from torch import Tensor as torchTensor

    _array = NDArray | JaxArray | torchTensor | tensorflowTensor
    Array = tp.TypeVar("Array", bound=_array)
    ArrayLike = tp.TypeVar("ArrayLike", bound=_array | float | int)

    Backend = tp.Literal["numpy", "numba", "jax", "torch", "tensorflow"] | None
