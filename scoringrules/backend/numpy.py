import typing as tp

import numpy as np
from scipy.special import (
    beta,
    betainc,
    comb,
    erf,
    expi,
    factorial,
    gamma,
    gammainc,
    gammaincc,
    hyp2f1,
    i0,
    i1,
)

if tp.TYPE_CHECKING:
    from numpy.typing import NDArray

    ArrayLike = NDArray | float

from .base import ArrayBackend

Dtype = tp.TypeVar("Dtype")


class NumpyBackend(ArrayBackend):
    """Numpy backend."""

    name = "numpy"

    def asarray(self, obj: "ArrayLike", /, *, dtype: Dtype | None = None) -> "NDArray":
        return np.asarray(obj, dtype=dtype)

    def broadcast_arrays(self, *arrays: "NDArray") -> tuple["NDArray", ...]:
        return np.broadcast_arrays(*arrays)

    def mean(
        self,
        x: "NDArray",
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> "NDArray":
        return np.mean(x, axis=axis, keepdims=keepdims)

    def max(
        self, x: "NDArray", axis: int | tuple[int, ...] | None, keepdims: bool = False
    ) -> "NDArray":
        return np.max(x, axis=axis, keepdims=keepdims)

    def moveaxis(
        self,
        x: "NDArray",
        /,
        source: tuple[int, ...] | int,
        destination: tuple[int, ...] | int,
    ) -> "NDArray":
        return np.moveaxis(x, source, destination)

    def sum(
        self,
        x: "NDArray",
        /,
        axis: int | tuple[int, ...] | None = None,
        *,
        dtype: Dtype | None = None,
        keepdims: bool = False,
    ) -> "NDArray":
        return np.sum(x, axis=axis, dtype=dtype, keepdims=keepdims)

    def unique_values(self, x: "NDArray") -> "NDArray":
        return np.unique(x)

    def concat(
        self,
        arrays: tuple["NDArray", ...] | list["NDArray"],
        /,
        *,
        axis: int | None = 0,
    ) -> "NDArray":
        return np.concatenate(arrays, axis=axis)

    def expand_dims(self, x: "NDArray", /, axis: int = 0) -> "NDArray":
        return np.expand_dims(x, axis=axis)

    def squeeze(
        self, x: "NDArray", /, *, axis: int | tuple[int, ...] | None = None
    ) -> "NDArray":
        return np.squeeze(x, axis=axis)

    def stack(
        self, arrays: tuple["NDArray", ...] | list["NDArray"], /, *, axis: int = 0
    ) -> "NDArray":
        return np.stack(arrays, axis=axis)

    def arange(
        self,
        start: int | float,
        /,
        stop: int | float | None = None,
        step: int | float = 1,
        *,
        dtype: Dtype | None = None,
    ) -> "NDArray":
        return np.arange(start, stop, step, dtype=dtype)

    def zeros(
        self, shape: int | tuple[int, ...], *, dtype: Dtype | None = None
    ) -> "NDArray":
        return np.zeros(shape, dtype=dtype)

    def abs(self, x: "NDArray") -> "NDArray":
        return np.abs(x)

    def exp(self, x: "NDArray") -> "NDArray":
        return np.exp(x)

    def isnan(self, x: "NDArray") -> "NDArray":
        return np.isnan(x)

    def log(self, x: "NDArray") -> "NDArray":
        return np.log(x)

    def sqrt(self, x: "NDArray") -> "NDArray":
        return np.sqrt(x)

    def any(
        self,
        x: "NDArray",
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> "NDArray":
        return np.any(x, axis=axis, keepdims=keepdims)

    def all(
        self,
        x: "NDArray",
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> "NDArray":
        return np.all(x, axis=axis, keepdims=keepdims)

    def sort(
        self,
        x: "NDArray",
        /,
        *,
        axis: int = -1,
        descending: bool = False,
    ) -> "NDArray":
        x = -x if descending else x
        out = np.sort(x, axis=axis)
        return -out if descending else out

    def norm(
        self, x: "NDArray", axis: int | tuple[int, ...] | None = None
    ) -> "NDArray":
        return np.linalg.norm(x, axis=axis)

    def erf(self, x):
        return erf(x)

    def apply_along_axis(self, func1d: tp.Callable, x: "NDArray", axis: int):
        return np.apply_along_axis(func1d, axis, x)

    def floor(self, x: "NDArray") -> "NDArray":
        return np.floor(x)

    def minimum(self, x: "NDArray", y: "ArrayLike") -> "NDArray":
        return np.minimum(x, y)

    def maximum(self, x: "NDArray", y: "ArrayLike") -> "NDArray":
        return np.maximum(x, y)

    def beta(self, x: "NDArray", y: "NDArray") -> "NDArray":
        return beta(x, y)

    def betainc(self, x: "NDArray", y: "NDArray", z: "NDArray") -> "NDArray":
        return betainc(x, y, z)

    def mbessel0(self, x: "NDArray") -> "NDArray":
        return i0(x)

    def mbessel1(self, x: "NDArray") -> "NDArray":
        return i1(x)

    def gamma(self, x: "NDArray") -> "NDArray":
        return gamma(x)

    def gammainc(self, x: "NDArray", y: "NDArray") -> "NDArray":
        return gammainc(x, y)

    def gammalinc(self, x: "NDArray", y: "NDArray") -> "NDArray":
        return gammainc(x, y) * gamma(x)

    def gammauinc(self, x: "NDArray", y: "NDArray") -> "NDArray":
        return gammaincc(x, y) * gamma(x)

    def factorial(self, n: "ArrayLike") -> "ArrayLike":
        return factorial(n)

    def hypergeometric(
        self, a: "NDArray", b: "NDArray", c: "NDArray", z: "NDArray"
    ) -> "NDArray":
        return hyp2f1(a, b, c, z)

    def comb(self, n: "ArrayLike", k: "ArrayLike") -> "ArrayLike":
        return comb(n, k)

    def expi(self, x: "NDArray") -> "NDArray":
        return expi(x)

    def where(self, condition: "NDArray", x1: "NDArray", x2: "NDArray") -> "NDArray":
        return np.where(condition, x1, x2)

    def size(self, x: "NDArray") -> int:
        return x.size

    def inv(self, x: "NDArray") -> "NDArray":
        return np.linalg.inv(x)

    def cov(self, x: "NDArray") -> "NDArray":
        # TODO: This deserves some axis parameter probably?
        if len(x.shape) == 2:
            return np.cov(x)
        else:
            centered = x - x.mean(axis=-2, keepdims=True)
            return (centered.transpose(0, 2, 1) @ centered) / (x.shape[-2] - 1)

    def det(self, x: "NDArray") -> "NDArray":
        return np.linalg.det(x)


class NumbaBackend(NumpyBackend):
    """Numba backend."""

    name = "numba"
