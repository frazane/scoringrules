import typing as tp
import math
from importlib import import_module

from .base import ArrayBackend

if tp.TYPE_CHECKING:
    import tensorflow as tf

    Tensor = tf.Tensor
    TensorLike = Tensor | bool | float | int
else:
    tf = None
Dtype = tp.TypeVar("Dtype")


class TensorflowBackend(ArrayBackend):
    """Tensorflow backend"""

    name = "tensorflow"

    def __init__(self) -> None:
        global tf
        if tf is None and not tp.TYPE_CHECKING:
            tf = import_module("tensorflow")

        self.pi = tf.constant(math.pi)

    def asarray(
        self,
        obj: "TensorLike",
        /,
        *,
        dtype: Dtype | None = None,
    ) -> "Tensor":
        return tf.convert_to_tensor(obj, dtype=dtype)

    def mean(
        self,
        x: "Tensor",
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> "Tensor":
        return tf.math.reduce_mean(x, axis=axis, keepdims=keepdims)

    def moveaxis(
        self,
        x: "Tensor",
        /,
        source: tuple[int, ...] | int,
        destination: tuple[int, ...] | int,
    ) -> "Tensor":
        return tf.experimental.numpy.moveaxis(x, source, destination)

    def sum(
        self,
        x: "Tensor",
        /,
        axis: int | tuple[int, ...] | None = None,
        *,
        keepdims: bool = False,
    ) -> "Tensor":
        return tf.math.reduce_sum(x, axis=axis, keepdims=keepdims)

    def unique_values(self, x: "Tensor", /) -> "Tensor":
        return tf.unique(x)

    def concat(
        self,
        arrays: tuple["Tensor", ...] | list["Tensor"],
        /,
        *,
        axis: int | None = 0,
    ) -> "Tensor":
        return tf.concat(arrays, axis=axis)

    def expand_dims(
        self, x: "Tensor", /, axis: int | tuple[int] = 0
    ) -> "Tensor":
        return tf.expand_dims(x, axis=axis)

    def squeeze(
        self, x: "Tensor", /, *, axis: int | tuple[int, ...] | None = None
    ) -> "Tensor":
        return tf.squeeze(x, axis=axis)

    def stack(
        self, arrays: tuple["Tensor", ...] | list["Tensor"], /, *, axis: int = 0
    ) -> "Tensor":
        return tf.stack(arrays, axis=axis)

    def arange(
        self,
        start: int | float,
        /,
        stop: int | float | None = None,
        step: int | float = 1,
        *,
        dtype: Dtype | None = None,
    ) -> "Tensor":
        return tf.range(start, stop, step, dtype=dtype)

    def zeros(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: Dtype | None = None,
    ) -> "Tensor":
        return tf.zeros(shape, dtype=dtype)

    def abs(self, x: "Tensor") -> "Tensor":
        return tf.math.abs(x)

    def exp(self, x: "Tensor") -> "Tensor":
        return tf.math.exp(x)

    def isnan(self, x: "Tensor") -> "Tensor":
        return tf.math.is_nan(x)

    def log(self, x: "Tensor") -> "Tensor":
        return tf.math.log(x)

    def sqrt(self, x: "Tensor") -> "Tensor":
        return tf.math.sqrt(x)

    def any(
        self,
        x: "Tensor",
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> "Tensor":
        return tf.math.reduce_any(x, axis=axis, keepdims=keepdims)

    def all(
        self,
        x: "Tensor",
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> "Tensor":
        return tf.math.reduce_all(x, axis=axis, keepdims=keepdims)

    def sort(
        self,
        x: "Tensor",
        /,
        *,
        axis: int = -1,
        descending: bool = False,
    ) -> "Tensor":
        direction = "DESCENDING" if descending else "ASCENDING"
        return tf.sort(x, axis=axis, direction=direction)[0]

    def norm(
        self, x: "Tensor", axis: int | tuple[int, ...] | None = None
    ) -> "Tensor":
        return tf.norm(x, axis=axis)

    def erf(self, x: "Tensor") -> "Tensor":
        return tf.math.erf(x)

    def apply_along_axis(
        self, func1d: tp.Callable[["Tensor"], "Tensor"], x: "Tensor", axis: int
    ):
        try:
            x_shape = list(tf.shape(x))
            flat = tf.map_fn(func1d, tf.reshape(x_shape.pop(axis), [-1]))
            return tf.reshape(flat, x_shape)
        except Exception:
            return tf.stack(
                [func1d(x_i) for x_i in tf.unstack(x, axis=axis)], axis=axis
            )


if __name__ == "__main__":
    B = TensorflowBackend()
    out = B.mean(tf.ones(10))
