"""A minimal subset of array API standard from
https://github.com/data-apis/array-api/tree/accf8c20d8fae9b760e59a40ab43b556fd8410cb/src/array_api_stubs/_2022_12
with few modifications. Just for static typing.
"""
from types import EllipsisType
from typing import TypeVar

from typing_extensions import Self

Dtype = TypeVar("Dtype")


class Array:
    """A minimal implementation of an array class following python array API standard."""

    def __init__(self: Self) -> None:
        ...

    @property
    def dtype(self: Self) -> Dtype:
        ...

    @property
    def ndim(self: Self) -> int:
        ...

    @property
    def shape(self: Self) -> tuple[int, ...]:
        ...

    @property
    def size(self: Self) -> int:
        ...

    @property
    def T(self: Self) -> Self:
        ...

    def __abs__(self: Self, /) -> Self:
        ...

    def __add__(self: Self, other: int | float | Self, /) -> Self:
        ...

    def __radd__(self: Self, other: int | float | Self, /) -> Self:
        ...

    def __and__(self: Self, other: int | bool | Self, /) -> Self:
        ...

    def __rand__(self: Self, other: int | bool | Self, /) -> Self:
        ...

    def __eq__(self: Self, other: int | float | bool | Self, /) -> Self:
        ...  # type: ignore

    def __req__(self: Self, other: int | float | bool | Self, /) -> Self:
        ...  # type: ignore

    def __float__(self: Self, /) -> float:
        ...

    def __floordiv__(self: Self, other: int | float | Self, /) -> Self:
        ...

    def __rfloordiv__(self: Self, other: int | float | Self, /) -> Self:
        ...

    def __ge__(self: Self, other: Self | int | float, /) -> Self:
        ...  # type: ignore

    def __rge__(self: Self, other: Self | int | float, /) -> Self:
        ...  # type: ignore

    def __getitem__(
        self: Self,
        key: int
        | slice
        | EllipsisType
        | tuple[int | slice | EllipsisType | None, ...]
        | Self
        | None,
        /,
    ) -> Self:
        ...

    def __gt__(self: Self, other: int | float | Self, /) -> Self:
        ...

    def __rgt__(self: Self, other: int | float | Self, /) -> Self:
        ...

    def __index__(self: Self, /) -> int:
        ...

    def __int__(self: Self, /) -> int:
        ...

    def __invert__(self: Self, /) -> Self:
        ...

    def __le__(self: Self, other: int | float | Self, /) -> Self:
        ...  # type: ignore

    def __rle__(self: Self, other: int | float | Self, /) -> Self:
        ...  # type: ignore

    def __lt__(self: Self, other: int | float | Self, /) -> Self:
        ...

    def __rlt__(self: Self, other: int | float | Self, /) -> Self:
        ...

    def __matmul__(self: Self, other: Self, /) -> Self:
        ...

    def __mod__(self: Self, other: int | float | Self, /) -> Self:
        ...

    def __rmod__(self: Self, other: int | float | Self, /) -> Self:
        ...

    def __mul__(self: Self, other: int | float | Self, /) -> Self:
        ...

    def __rmul__(self: Self, other: int | float | Self, /) -> Self:
        ...

    def __ne__(self: Self, other: int | float | bool | Self, /) -> Self:
        ...  # type: ignore

    def __rne__(self: Self, other: int | float | bool | Self, /) -> Self:
        ...  # type: ignore

    def __neg__(self: Self, /) -> Self:
        ...

    def __or__(self: Self, other: int | bool | Self, /) -> Self:
        ...

    def __ror__(self: Self, other: int | bool | Self, /) -> Self:
        ...

    def __pos__(self: Self, /) -> Self:
        ...

    def __pow__(self: Self, other: int | float | Self, /) -> Self:
        ...

    def __rpow__(self: Self, other: int | float | Self, /) -> Self:
        ...

    def __setitem__(
        self: Self,
        key: int | slice | EllipsisType | tuple[int | slice | EllipsisType, ...] | Self,
        value: int | float | bool | Self,
        /,
    ) -> None:
        ...

    def __sub__(self: Self, other: int | float | Self, /) -> Self:
        ...

    def __rsub__(self: Self, other: int | float | Self, /) -> Self:
        ...

    def __truediv__(self: Self, other: int | float | Self, /) -> Self:
        ...

    def __rtruediv__(self: Self, other: int | float | Self, /) -> Self:
        ...


__all__ = ["Array", "Dtype"]
