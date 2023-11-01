import typing as tp
from importlib import import_module

if tp.TYPE_CHECKING:
    import jax
    import jax.numpy as jnp
    import jax.scipy as jsp
    from jax import Array

    ArrayLike = Array | float | int
else:
    jnp = None
    jsp = None
    jax = None


from .base import ArrayBackend

Dtype = tp.TypeVar("Dtype")


class JaxBackend(ArrayBackend):
    """Jax backend."""

    name = "jax"

    def __init__(self):
        global jnp, jsp, jax
        if jnp is None and jsp is None and not tp.TYPE_CHECKING:
            jax = import_module("jax")
            jnp = import_module("jax.numpy")
            jsp = import_module("jax.scipy")

    def asarray(self, obj: "ArrayLike", /, *, dtype: Dtype | None = None) -> "Array":
        return jnp.asarray(obj, dtype=dtype)

    def mean(
        self,
        x: "Array",
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> "Array":
        return jnp.mean(x, axis=axis, keepdims=keepdims)

    def moveaxis(
        self,
        x: "Array",
        /,
        source: tuple[int, ...] | int,
        destination: tuple[int, ...] | int,
    ) -> "Array":
        return jnp.moveaxis(x, source, destination)

    def sum(
        self,
        x: "Array",
        /,
        axis: int | tuple[int, ...] | None = None,
        *,
        dtype: Dtype | None = None,
        keepdims: bool = False,
    ) -> "Array":
        return jnp.sum(x, axis=axis, dtype=dtype, keepdims=keepdims)

    def unique_values(self, x: "Array") -> "Array":
        return jnp.unique(x)

    def concat(
        self, arrays: tuple["Array", ...] | list["Array"], /, *, axis: int | None = 0
    ) -> "Array":
        return jnp.concatenate(arrays, axis=axis)

    def expand_dims(self, x: "Array", /, axis: int = 0) -> "Array":
        return jnp.expand_dims(x, axis=axis)

    def squeeze(
        self, x: "Array", /, *, axis: int | tuple[int, ...] | None = None
    ) -> "Array":
        return jnp.squeeze(x, axis=axis)

    def stack(
        self, arrays: tuple["Array", ...] | list["Array"], /, *, axis: int = 0
    ) -> "Array":
        return jnp.stack(arrays, axis=axis)

    def arange(
        self,
        start: int | float,
        /,
        stop: int | float | None = None,
        step: int | float = 1,
        *,
        dtype: Dtype | None = None,
    ) -> "Array":
        return jnp.arange(start, stop, step, dtype=dtype)

    def zeros(
        self, shape: int | tuple[int, ...], *, dtype: Dtype | None = None
    ) -> "Array":
        return jnp.zeros(shape, dtype=dtype)

    def abs(self, x: "Array") -> "Array":
        return jnp.abs(x)

    def exp(self, x: "Array") -> "Array":
        return jnp.exp(x)

    def isnan(self, x: "Array") -> "Array":
        return jnp.isnan(x)

    def log(self, x: "Array") -> "Array":
        return jnp.log(x)

    def sqrt(self, x: "Array") -> "Array":
        return jnp.sqrt(x)

    def any(
        self,
        x: "Array",
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> "Array":
        return jnp.any(x, axis=axis, keepdims=keepdims)

    def all(
        self,
        x: "Array",
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> "Array":
        return jnp.all(x, axis=axis, keepdims=keepdims)

    def sort(
        self,
        x: "Array",
        /,
        *,
        axis: int = -1,
        descending: bool = False,
    ) -> "Array":
        x = -x if descending else x
        out = jnp.sort(x, axis=axis)  # TODO: this is slow! why?
        return -out if descending else out

    def norm(self, x: "Array", axis: int | tuple[int, ...] | None = None) -> "Array":
        return jnp.linalg.norm(x, axis=axis)

    def erf(self, x: "Array") -> "Array":
        return jsp.special.erf(x)

    def apply_along_axis(
        self, func1d: tp.Callable[["Array"], "Array"], x: "Array", axis: int
    ):
        try:
            x_shape = list(x.shape)
            return jax.vmap(func1d)(x.reshape(-1, x_shape.pop(axis))).reshape(x_shape)
        except Exception:
            return jnp.apply_along_axis(func1d, axis, x)


if __name__ == "__main__":
    B = JaxBackend()
    out = B.mean(jnp.ones(10))
