import typing as tp
from importlib import import_module

from .base import ArrayBackend

if tp.TYPE_CHECKING:
    import torch

    Tensor = torch.Tensor
else:
    torch = None

Dtype = tp.TypeVar("Dtype")


class TorchBackend(ArrayBackend[Tensor]):
    """Torch backend."""

    pi: torch.asarray(torch.pi)

    def __init__(self) -> None:
        global torch
        if torch is None and not tp.TYPE_CHECKING:
            torch = import_module("torch")

    def asarray(
        self,
        obj: Tensor | bool | int | float,
        /,
        *,
        dtype: Dtype | None = None,
    ) -> Tensor:
        return torch.asarray(obj, dtype=dtype)

    def mean(
        self,
        x: Tensor,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Tensor:
        return torch.mean(x, axis=axis, keepdim=keepdims)

    def moveaxis(
        self,
        x: Tensor,
        /,
        source: tuple[int, ...] | int,
        destination: tuple[int, ...] | int,
    ) -> Tensor:
        return torch.moveaxis(x, source, destination)

    def sum(
        self,
        x: Tensor,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Tensor:
        return torch.sum(x, axis=axis, keepdim=keepdims)

    def unique_values(self, x: Tensor, /) -> Tensor:
        return torch.unique(x)

    def concat(
        self, arrays: tuple[Tensor, ...] | list[Tensor], /, *, axis: int | None = 0
    ) -> Tensor:
        return torch.concat(arrays, axis=axis)

    def expand_dims(self, x: Tensor, /, *, axis: int = 0) -> Tensor:
        return torch.unsqueeze(x, dim=axis)

    def squeeze(
        self, x: Tensor, /, *, axis: int | tuple[int, ...] | None = None
    ) -> Tensor:
        return torch.squeeze(x, dim=axis)

    def stack(
        self, arrays: tuple[Tensor, ...] | list[Tensor], /, *, axis: int = 0
    ) -> Tensor:
        return torch.stack(arrays, dim=axis)

    def arange(
        self,
        start: int | float,
        /,
        stop: int | float | None = None,
        step: int | float = 1,
        *,
        dtype: Dtype | None = None,
    ) -> Tensor:
        return torch.arange(start, stop, step, dtype=dtype)

    def zeros(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: Dtype | None = None,
    ) -> Tensor:
        return torch.zeros(shape, dtype=dtype)

    def abs(self, x: Tensor) -> Tensor:
        return torch.abs(x)

    def exp(self, x: Tensor) -> Tensor:
        return torch.exp(x)

    def isnan(self, x: Tensor) -> Tensor:
        return torch.isnan(x)

    def log(self, x: Tensor) -> Tensor:
        return torch.log(x)

    def sqrt(self, x: Tensor) -> Tensor:
        return torch.sqrt(x)

    def any(
        self,
        x: Tensor,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Tensor:
        return torch.any(x, dim=axis, keepdim=keepdims)

    def all(
        self,
        x: Tensor,
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> Tensor:
        return torch.all(x, dim=axis, keepdim=keepdims)

    def sort(
        self,
        x: Tensor,
        /,
        *,
        axis: int = -1,
        descending: bool = False,
        stable: bool = True,
    ) -> Tensor:
        return torch.sort(x, stable=stable, dim=axis, descending=descending)

    def norm(self, x: Tensor, axis: int | tuple[int, ...] | None = None) -> Tensor:
        return torch.norm(x, dim=axis)

    def erf(self, x: Tensor) -> Tensor:
        return torch.special.erf(x)

if __name__ == "__main__":
    B = TorchBackend()
    out = B.mean(torch.ones(10))
