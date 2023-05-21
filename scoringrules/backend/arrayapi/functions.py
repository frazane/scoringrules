"""A minimal subset of array API standard functions from
https://github.com/data-apis/array-api/tree/accf8c20d8fae9b760e59a40ab43b556fd8410cb/src/array_api_stubs/_2022_12
with few modifications. Just for static typing.
"""
from typing import Literal, Protocol, TypeVar

from array_type import Array, Dtype

device = TypeVar("device")
SupportsBufferProtocol = TypeVar("SupportsBufferProtocol")

inf = float("inf")


class ArrayAPIConstantsProtocol(Protocol):
    e: float = 2.718281828459045
    inf = float("inf")
    nan = float("nan")
    newaxis = None
    pi: float = 3.141592653589793


class ArrayAPILinalgProtocol(Protocol):
    @staticmethod
    def norm(
        x: Array,
        /,
        *,
        keepdims: bool = False,
        ord: int | float | Literal["fro", "nuc"] = "fro",  # type: ignore
    ) -> Array:
        """Computes the matrix norm of a matrix (or a stack of matrices) ``x``."""


class ArrayAPIProtocol(ArrayAPIConstantsProtocol, Protocol):
    """The python array API standard implemented as Protocol for static typing."""

    linalg = ArrayAPILinalgProtocol

    @staticmethod
    def asarray(
        obj: Array | bool | int | float,
        /,
        *,
        dtype: Dtype | None = None,
        device: device | None = None,
        copy: bool | None = None,
    ) -> Array:
        """Convert the input to an array."""

    @staticmethod
    def where(condition: Array, x1: Array, x2: Array, /) -> Array:
        """Return elements chosen from ``x1`` or ``x2`` depending on ``condition``."""

    @staticmethod
    def mean(
        x: Array,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array:
        """Calculate the arithmetic mean of the input array ``x``."""

    @staticmethod
    def std(
        x: Array,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        correction: int | float = 0.0,
        keepdims: bool = False,
    ) -> Array:
        """Calculate the standard deviation of the input array ``x``."""

    @staticmethod
    def moveaxis(
        x: Array, /, source: tuple[int, ...] | int, destination: tuple[int, ...] | int
    ) -> Array:
        """Permutes the axes (dimensions) of an array ``x``."""

    @staticmethod
    def sum(
        x: Array,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: Dtype | None = None,
        keepdims: bool = False,
    ) -> Array:
        """Calculate the sum of the input array ``x``."""

    # NOTE: this should be unique_values
    @staticmethod
    def unique(x: Array, /) -> Array:
        """Return the unique elements of an input array ``x``."""

    @staticmethod
    def concat(
        arrays: tuple[Array, ...] | list[Array], /, *, axis: int | None = 0
    ) -> Array:
        """Join a sequence of arrays along an existing axis."""

    @staticmethod
    def expand_dims(x: Array, /, *, axis: int = 0) -> Array:
        """Expand the shape of an array by inserting a new axis (dimension) of size one at the position specified by ``axis``."""

    @staticmethod
    def reshape(
        x: Array, /, shape: tuple[int, ...], *, copy: bool | None = None
    ) -> Array:
        """Reshape an array without changing its data."""

    @staticmethod
    def squeeze(x: Array, /, *, axis: int | tuple[int, ...] | None = None) -> Array:
        """Remove singleton dimensions (axes) from ``x``."""

    @staticmethod
    def stack(arrays: tuple[Array, ...] | list[Array], /, *, axis: int = 0) -> Array:
        """Join a sequence of arrays along a new axis."""

    @staticmethod
    def arange(
        start: int | float,
        /,
        stop: int | float | None = None,
        step: int | float = 1,
        *,
        dtype: Dtype | None = None,
        device: device | None = None,
    ) -> Array:
        """Return evenly spaced values within the half-open interval ``[start, stop)`` as a one-dimensional array."""

    @staticmethod
    def eye(
        n_rows: int,
        n_cols: int | None = None,
        /,
        *,
        k: int = 0,
        dtype: Dtype | None = None,
        device: device | None = None,
    ) -> Array:
        """Return a two-dimensional array with ones on the ``k``\\th diagonal and zeros elsewhere."""

    @staticmethod
    def zeros(
        shape: int | tuple[int, ...],
        *,
        dtype: Dtype | None = None,
        device: device | None = None,
    ) -> Array:
        """Return a new array having a specified ``shape`` and filled with zeros."""

    @staticmethod
    def zeros_like(
        x: Array, /, *, dtype: Dtype | None = None, device: device | None = None
    ) -> Array:
        """Return a new array filled with zeros and having the same ``shape`` as an input array ``x``."""

    @staticmethod
    def ones_like(
        x: Array, /, *, dtype: Dtype | None = None, device: device | None = None
    ) -> Array:
        """Return a new array filled with ones and having the same ``shape`` as an input array ``x``."""

    @staticmethod
    def abs(x: Array, /) -> Array:
        """Calculate the absolute value for each element ``x_i`` of the input array ``x``."""

    @staticmethod
    def exp(x: Array, /) -> Array:
        """Calculate an implementation-dependent approximation to the exponential function for each element ``x_i`` of the input array ``x`` (``e`` raised to the power of ``x_i``, where ``e`` is the base of the natural logarithm)."""

    @staticmethod
    def isnan(x: Array, /) -> Array:
        """Test each element ``x_i`` of the input array ``x`` to determine whether the element is ``NaN``."""

    @staticmethod
    def log(x: Array, /) -> Array:
        """Calculate an implementation-dependent approximation to the natural (base ``e``) logarithm for each element ``x_i`` of the input array ``x``."""

    @staticmethod
    def pow(x1: Array, x2: Array, /) -> Array:
        """Calculate an implementation-dependent approximation of exponentiation by raising each element ``x1_i`` (the base) of the input array ``x1`` to the power of ``x2_i`` (the exponent), where ``x2_i`` is the corresponding element of the input array ``x2``."""

    @staticmethod
    def sqrt(x: Array, /) -> Array:
        """Calculate the principal square root for each element ``x_i`` of the input array ``x``."""

    @staticmethod
    def all(
        x: Array,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array:
        """Test whether all input array elements evaluate to ``True`` along a specified axis."""

    @staticmethod
    def any(
        x: Array,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Array:
        """Test whether any input array element evaluates to ``True`` along a specified axis."""

    @staticmethod
    def argsort(
        x: Array, /, *, axis: int = -1, descending: bool = False, stable: bool = True
    ) -> Array:
        """Return the indices that sort an array ``x`` along a specified axis."""

    @staticmethod
    def sort(
        x: Array, /, *, axis: int = -1, descending: bool = False, stable: bool = True
    ) -> Array:
        """Return a sorted copy of an input array ``x``."""
