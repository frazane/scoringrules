import typing as tp

if tp.TYPE_CHECKING:
    from torch import Tensor
    from jax import Array as JaxArray
    from numpy.typing import ArrayLike, NDArray

    _array = NDArray | JaxArray | Tensor
    Array = tp.TypeVar("Array", bound=_array)
    ArrayLike = tp.TypeVar("ArrayLike", bound=_array | float)