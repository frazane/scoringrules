import math
import typing as tp
from importlib import import_module

from .base import ArrayBackend

if tp.TYPE_CHECKING:
    import tensorflow as tf

    Tensor = tf.Tensor
    TensorLike = Tensor | bool | float | int
    DTYPE = tf.float32
else:
    tf = None
    DTYPE = None
Dtype = tp.TypeVar("Dtype")


class TensorflowBackend(ArrayBackend):
    """Tensorflow backend."""

    name = "tensorflow"

    def __init__(self) -> None:
        global tf, DTYPE
        if tf is None and not tp.TYPE_CHECKING:
            tf = import_module("tensorflow")
            DTYPE = tf.float32
        self.pi = tf.constant(math.pi, dtype=DTYPE)

    def asarray(
        self,
        obj: "TensorLike",
        /,
        *,
        dtype: Dtype | None = None,
    ) -> "Tensor":
        if dtype is None:
            dtype = DTYPE
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

    def max(
        self, x: "Tensor", axis: int | tuple[int, ...] | None, keepdims: bool = False
    ) -> "Tensor":
        return tf.math.reduce_max(x, axis=axis, keepdims=keepdims)

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

    # tf.expand_dims() doesn't support tuples in v2.15
    def expand_dims(self, x: "Tensor", /, axis: int | tuple[int] = 0) -> "Tensor":
        if isinstance(axis, int):
            return tf.expand_dims(x, axis=axis)
        elif isinstance(axis, tuple | list):
            out_ndim = len(axis) + x.ndim
            axis = [a + out_ndim if a < 0 else a for a in axis]
            shape_it = iter(x.shape)
            shape = [1 if ax in axis else next(shape_it) for ax in range(out_ndim)]
            return tf.reshape(x, shape=shape)

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
        if dtype is None:
            dtype = DTYPE
        return tf.range(start, stop, step, dtype=dtype)

    def zeros(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: Dtype | None = None,
    ) -> "Tensor":
        if dtype is None:
            dtype = DTYPE
        return tf.zeros(shape, dtype=dtype)

    def abs(self, x: "Tensor") -> "Tensor":
        return tf.math.abs(x)

    def exp(self, x: "Tensor") -> "Tensor":
        return tf.math.exp(x)

    def isnan(
        self,
        x: "Tensor",
        *,
        dtype: Dtype | None = None,
    ) -> "Tensor":
        if dtype is None:
            dtype = DTYPE
        return tf.cast(tf.math.is_nan(x), dtype=dtype)

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
        dtype: Dtype | None = None,
    ) -> "Tensor":
        if dtype is None:
            dtype = DTYPE
        return tf.cast(tf.math.reduce_any(x, axis=axis, keepdims=keepdims), dtype=dtype)

    def all(
        self,
        x: "Tensor",
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        dtype: Dtype | None = None,
    ) -> "Tensor":
        if dtype is None:
            dtype = DTYPE
        return tf.cast(tf.math.reduce_all(x, axis=axis, keepdims=keepdims), dtype=dtype)

    def sort(
        self,
        x: "Tensor",
        /,
        *,
        axis: int = -1,
        descending: bool = False,
    ) -> "Tensor":
        direction = "DESCENDING" if descending else "ASCENDING"
        return tf.sort(x, axis=axis, direction=direction)

    def norm(self, x: "Tensor", axis: int | tuple[int, ...] | None = None) -> "Tensor":
        return tf.norm(x, axis=axis)

    def erf(self, x: "Tensor") -> "Tensor":
        return tf.math.erf(x)

    def apply_along_axis(
        self, func1d: tp.Callable[["Tensor"], "Tensor"], x: "Tensor", axis: int
    ):
        try:
            x_shape = x.shape.as_list()
            flat = tf.map_fn(func1d, tf.reshape(x, [-1, x_shape.pop(axis)]))
            return tf.reshape(flat, x_shape)
        except Exception:
            return tf.stack(
                [func1d(x_i) for x_i in tf.unstack(x, axis=axis)], axis=axis
            )

    def floor(self, x: "Tensor") -> "Tensor":
        return tf.math.floor(x)

    def minimum(self, x: "Tensor", y: "TensorLike") -> "Tensor":
        return tf.math.minimum(x, y)

    def maximum(self, x: "Tensor", y: "TensorLike") -> "Tensor":
        return tf.math.maximum(x, y)

    def beta(self, x: "Tensor", y: "Tensor") -> "Tensor":
        return tf.math.exp(
            tf.math.lgamma(x) + tf.math.lgamma(y) - tf.math.lgamma(x + y)
        )

    def betainc(self, x: "Tensor", y: "Tensor", z: "Tensor") -> "Tensor":
        return tf.math.betainc(x, y, z)

    def mbessel0(self, x: "Tensor") -> "Tensor":
        return tf.math.bessel_i0e(x)

    def mbessel1(self, x: "Tensor") -> "Tensor":
        return tf.math.bessel_i1e(x)

    def gamma(self, x: "Tensor") -> "Tensor":
        return tf.math.exp(tf.math.lgamma(x))

    def gammainc(self, x: "Tensor", y: "Tensor") -> "Tensor":
        return tf.math.igamma(x, y)

    def gammalinc(self, x: "Tensor", y: "Tensor") -> "Tensor":
        return tf.math.igamma(x, y) * tf.math.exp(tf.math.lgamma(x))

    def gammauinc(self, x: "Tensor", y: "Tensor") -> "Tensor":
        return tf.math.igammac(x, y) * tf.math.exp(tf.math.lgamma(x))

    def factorial(self, n: "TensorLike") -> "TensorLike":
        return tf.math.exp(tf.math.lgamma(n + 1))

    def hypergeometric(
        self, a: "Tensor", b: "Tensor", c: "Tensor", z: "Tensor"
    ) -> "Tensor":
        raise NotImplementedError

    def comb(self, n: "Tensor", k: "Tensor") -> "Tensor":
        return self.factorial(n) // (self.factorial(k) * self.factorial(n - k))

    def expi(self, x: "Tensor") -> "Tensor":
        return tf.math.special.expint(x)

    def where(self, condition: "Tensor", x1: "Tensor", x2: "Tensor") -> "Tensor":
        return tf.where(condition, x1, x2)


if __name__ == "__main__":
    B = TensorflowBackend()
    out = B.mean(tf.ones(10))
