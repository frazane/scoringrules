"""A minimal subset of array API standard functions from
https://github.com/data-apis/array-api/tree/accf8c20d8fae9b760e59a40ab43b556fd8410cb/src/array_api_stubs/_2022_12
with few modifications.
"""
import abc
import typing as tp
import torch

Array = tp.TypeVar("Array")
Dtype = tp.TypeVar("Dtype")


class ArrayBackend(abc.ABC):
    """The python array API standard implemented as Protocol for static typing."""

    name: str
    e: float = 2.718281828459045
    inf = float("inf")
    nan = float("nan")
    newaxis = None
    pi: float = 3.141592653589793

    @abc.abstractmethod
    def asarray(
        self,
        obj: Array | bool | int | float,
        /,
        *,
        dtype: Dtype | None = None,
    ) -> Array:
        """Convert the input to an array."""

    @abc.abstractmethod
    def mean(
        self,
        x: Array,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array:
        """Calculate the arithmetic mean of the input array ``x``."""

    @abc.abstractmethod
    def moveaxis(
        self,
        x: Array,
        /,
        source: tuple[int, ...] | int,
        destination: tuple[int, ...] | int,
    ) -> Array:
        """Permutes the axes (dimensions) of an array ``x``."""

    @abc.abstractmethod
    def sum(
        self,
        x: Array,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: Dtype | None = None,
        keepdims: bool = False,
    ) -> Array:
        """Calculate the sum of the input array ``x``."""

    @abc.abstractmethod
    def unique_values(self, x: Array, /) -> Array:
        """Return the unique elements of an input array ``x``."""

    @abc.abstractmethod
    def concat(
        self, arrays: tuple[Array, ...] | list[Array], /, *, axis: int | None = 0
    ) -> Array:
        """Join a sequence of arrays along an existing axis."""

    @abc.abstractmethod
    def expand_dims(self, x: Array, /, *, axis: int = 0) -> Array:
        """Expand the shape of an array by inserting a new axis (dimension) of size one at the position specified by ``axis``."""

    @abc.abstractmethod
    def squeeze(
        self, x: Array, /, *, axis: int | tuple[int, ...] | None = None
    ) -> Array:
        """Remove singleton dimensions (axes) from ``x``."""

    @abc.abstractmethod
    def stack(
        self, arrays: tuple[Array, ...] | list[Array], /, *, axis: int = 0
    ) -> Array:
        """Join a sequence of arrays along a new axis."""

    @abc.abstractmethod
    def arange(
        self,
        start: int | float,
        /,
        stop: int | float | None = None,
        step: int | float = 1,
        *,
        dtype: Dtype | None = None,
    ) -> Array:
        """Return evenly spaced values within the half-open interval ``[start, stop)`` as a one-dimensional array."""

    @abc.abstractmethod
    def zeros(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: Dtype | None = None,
    ) -> Array:
        """Return a new array having a specified ``shape`` and filled with zeros."""

    @abc.abstractmethod
    def abs(self, x: Array, /) -> Array:
        """Calculate the absolute value for each element ``x_i`` of the input array ``x``."""

    @abc.abstractmethod
    def exp(self, x: Array, /) -> Array:
        """Calculate an implementation-dependent approximation to the exponential function for each element ``x_i`` of the input array ``x`` (``e`` raised to the power of ``x_i``, where ``e`` is the base of the natural logarithm)."""

    @abc.abstractmethod
    def isnan(self, x: Array, /) -> Array:
        """Test each element ``x_i`` of the input array ``x`` to determine whether the element is ``NaN``."""

    @abc.abstractmethod
    def log(self, x: Array, /) -> Array:
        """Calculate an implementation-dependent approximation to the natural (base ``e``) logarithm for each element ``x_i`` of the input array ``x``."""

    @abc.abstractmethod
    def sqrt(self, x: Array, /) -> Array:
        """Calculate the principal square root for each element ``x_i`` of the input array ``x``."""

    @abc.abstractmethod
    def all(
        self,
        x: Array,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array:
        """Test whether all input array elements evaluate to ``True`` along a specified axis."""

    @abc.abstractmethod
    def any(
        self,
        x: Array,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array:
        """Test whether any input array element evaluates to ``True`` along a specified axis."""

    @abc.abstractmethod
    def sort(
        self,
        x: Array,
        /,
        *,
        axis: int = -1,
        descending: bool = False,
        stable: bool = True,
    ) -> Array:
        """Return a sorted copy of an input array ``x``."""

    @abc.abstractmethod
    def norm(self, x: Array) -> Array:
        """Computes the matrix norm of a matrix (or a stack of matrices) ``x``."""

    @abc.abstractmethod
    def erf(self, x: Array) -> Array:
        """ERF"""


if __name__ == "__main__":
    B = ArrayBackend()
