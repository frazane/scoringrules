import typing as tp

if tp.TYPE_CHECKING:
    from jax import Array as JaxArray
    from numpy.typing import NDArray
    from torch import Tensor as torchTensor
    from tensorflow import Tensor as tensorflowTensor

    _array = NDArray | JaxArray | torchTensor | tensorflowTensor
    Array = tp.TypeVar("Array", bound=_array)
    ArrayLike = tp.TypeVar("ArrayLike", bound=_array | float | int)
