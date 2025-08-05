import abc
import typing as tp

if tp.TYPE_CHECKING:
    from scoringrules.core.typing import Array, ArrayLike


Dtype = tp.TypeVar("Dtype")


class ArrayBackend(abc.ABC):
    """The abstract class for array backends."""

    name: str
    e: float = 2.718281828459045
    inf = float("inf")
    nan = float("nan")
    newaxis = None
    pi: float = 3.141592653589793

    @abc.abstractmethod
    def asarray(
        self,
        obj: "ArrayLike",
        /,
        *,
        dtype: Dtype | None = None,
    ) -> "Array":
        """Convert the input to an array."""

    @abc.abstractmethod
    def broadcast_arrays(*arrays: "Array") -> tuple["Array", ...]:
        """Broadcast any number of arrays against each other."""

    @abc.abstractmethod
    def mean(
        self,
        x: "Array",
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> "Array":
        """Calculate the arithmetic mean of the input array ``x``."""

    @abc.abstractmethod
    def std(
        self,
        x: "Array",
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        bias: bool = False,
        keepdims: bool = False,
    ) -> "Array":
        """Calculate the standard deviation of the input array ``x``."""

    @abc.abstractmethod
    def quantile(
        self,
        x: "Array",
        q: "ArrayLike",
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> "Array":
        """Calculate the ``q`` quantiles of the input array ``x``."""

    @abc.abstractmethod
    def max(
        self,
        x: "Array",
        axis: int | tuple[int, ...] | None,
        keepdims: bool = False,
    ) -> "Array":
        """Return the maximum value of an input array ``x``."""

    @abc.abstractmethod
    def moveaxis(
        self,
        x: "Array",
        /,
        source: tuple[int, ...] | int,
        destination: tuple[int, ...] | int,
    ) -> "Array":
        """Permutes the axes (dimensions) of an array ``x``."""

    @abc.abstractmethod
    def sum(
        self,
        x: "Array",
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: Dtype | None = None,
        keepdims: bool = False,
    ) -> "Array":
        """Calculate the sum of the input array ``x``."""

    @abc.abstractmethod
    def cumsum(
        self,
        x: "Array",
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        dtype: Dtype | None = None,
    ) -> "Array":
        """Calculate the cumulative sum of the input array ``x``."""

    @abc.abstractmethod
    def unique_values(self, x: "Array", /) -> "Array":
        """Return the unique elements of an input array ``x``."""

    @abc.abstractmethod
    def concat(
        self,
        arrays: tuple["Array", ...] | list["Array"],
        /,
        *,
        axis: int | None = 0,
    ) -> "Array":
        """Join a sequence of arrays along an existing axis."""

    @abc.abstractmethod
    def expand_dims(self, x: "Array", /, axis: int = 0) -> "Array":
        """Expand the shape of an array by inserting a new axis (dimension) of size one at the position specified by ``axis``."""

    @abc.abstractmethod
    def squeeze(
        self, x: "Array", /, *, axis: int | tuple[int, ...] | None = None
    ) -> "Array":
        """Remove singleton dimensions (axes) from ``x``."""

    @abc.abstractmethod
    def stack(
        self, arrays: tuple["Array", ...] | list["Array"], /, *, axis: int = 0
    ) -> "Array":
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
    ) -> "Array":
        """Return evenly spaced values within the half-open interval ``[start, stop)`` as a one-dimensional array."""

    @abc.abstractmethod
    def zeros(
        self,
        shape: int | tuple[int, ...],
        *,
        dtype: Dtype | None = None,
    ) -> "Array":
        """Return a new array having a specified ``shape`` and filled with zeros."""

    @abc.abstractmethod
    def abs(self, x: "Array", /) -> "Array":
        """Calculate the absolute value for each element ``x_i`` of the input array ``x``."""

    @abc.abstractmethod
    def exp(self, x: "Array", /) -> "Array":
        """Calculate an implementation-dependent approximation to the exponential function for each element ``x_i`` of the input array ``x`` (``e`` raised to the power of ``x_i``, where ``e`` is the base of the natural logarithm)."""

    @abc.abstractmethod
    def isnan(self, x: "Array", /) -> "Array":
        """Test each element ``x_i`` of the input array ``x`` to determine whether the element is ``NaN``."""

    @abc.abstractmethod
    def log(self, x: "Array", /) -> "Array":
        """Calculate an implementation-dependent approximation to the natural (base ``e``) logarithm for each element ``x_i`` of the input array ``x``."""

    @abc.abstractmethod
    def sqrt(self, x: "Array", /) -> "Array":
        """Calculate the principal square root for each element ``x_i`` of the input array ``x``."""

    @abc.abstractmethod
    def all(
        self,
        x: "Array",
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> "Array":
        """Test whether all input array elements evaluate to ``True`` along a specified axis."""

    @abc.abstractmethod
    def any(
        self,
        x: "Array",
        /,
        *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
    ) -> "Array":
        """Test whether any input array element evaluates to ``True`` along a specified axis."""

    @abc.abstractmethod
    def sort(
        self,
        x: "Array",
        /,
        *,
        axis: int = -1,
        descending: bool = False,
    ) -> "Array":
        """Return a sorted copy of an input array ``x``."""

    @abc.abstractmethod
    def norm(self, x: "Array") -> "Array":
        """Compute the matrix norm of a matrix (or a stack of matrices) ``x``."""

    @abc.abstractmethod
    def erf(self, x: "Array") -> "Array":
        """Return the error function."""

    @abc.abstractmethod
    def apply_along_axis(
        self, func1d: tp.Callable[["Array"], "Array"], x: "Array", axis: int
    ) -> "Array":
        """Apply a function along a given axis of the input array."""

    @abc.abstractmethod
    def floor(self, x: "Array", /) -> "Array":
        """Calculate the integer component of each element ``x_i`` of the input array ``x``."""

    @abc.abstractmethod
    def minimum(self, x: "Array", y: "ArrayLike", /) -> "Array":
        """Calculate the minimum of each element ``x_i`` of the input array ``x`` with the value ``y``."""

    @abc.abstractmethod
    def maximum(self, x: "Array", y: "ArrayLike", /) -> "Array":
        """Calculate the maximum of each element ``x_i`` of the input array ``x`` with the value ``y``."""

    @abc.abstractmethod
    def beta(self, x: "Array", y: "Array", /) -> "Array":
        """Calculate the beta function at each element ``x_i`` of the input array ``x``."""

    @abc.abstractmethod
    def betainc(self, x: "Array", y: "Array", z: "Array", /) -> "Array":
        """Calculate the regularised incomplete beta function at each element ``x_i`` of the input array ``x``."""

    @abc.abstractmethod
    def mbessel0(self, x: "Array", /) -> "Array":
        """Calculate the modified Bessel function of the first kind of order 0 at each element ``x_i`` of the input array ``x``."""

    @abc.abstractmethod
    def mbessel1(self, x: "Array", /) -> "Array":
        """Calculate the modified Bessel function of the first kind of order 1 at each element ``x_i`` of the input array ``x``."""

    @abc.abstractmethod
    def gamma(self, x: "Array", /) -> "Array":
        """Calculate the gamma function at each element ``x_i`` of the input array ``x``."""

    @abc.abstractmethod
    def gammainc(self, x: "Array", y: "Array", /) -> "Array":
        """Calculate the regularised lower incomplete gamma function at each element ``x_i`` of the input array ``x``."""

    @abc.abstractmethod
    def gammalinc(self, x: "Array", y: "Array", /) -> "Array":
        """Calculate the lower incomplete gamma function at each element ``x_i`` of the input array ``x``."""

    @abc.abstractmethod
    def gammauinc(self, x: "Array", y: "Array", /) -> "Array":
        """Calculate the upper incomplete gamma function at each element ``x_i`` of the input array ``x``."""

    @abc.abstractmethod
    def factorial(self, n: "ArrayLike", /) -> "ArrayLike":
        """Calculate the factorial of the integer ``n``."""

    @abc.abstractmethod
    def hypergeometric(self, a: "Array", b: "Array", c: "Array", z: "Array") -> "Array":
        """Calculate the hypergeometric function at each element of the inputs ``a``, ``b``, ``c``, and ``z``."""

    @abc.abstractmethod
    def comb(self, n: "ArrayLike", k: "ArrayLike", /) -> "ArrayLike":
        """Calculate ``n`` choose ``k``."""

    @abc.abstractmethod
    def expi(self, x: "Array", /) -> "Array":
        """Calculate the exponential integral at each element ``x_i`` of the input array ``x``."""

    @abc.abstractmethod
    def where(self, condition: "Array", x1: "Array", x2: "Array", /) -> "Array":
        """Return elements chosen from x1 or x2 depending on condition."""

    @abc.abstractmethod
    def size(self, x: "Array") -> int:
        """Return the number of elements in the input array."""

    @abc.abstractmethod
    def indices(self, x: "Array") -> int:
        """Return an array representing the indices of a grid."""

    @abc.abstractmethod
    def inv(self, x: "Array") -> "Array":
        """Return the inverse of a matrix."""

    @abc.abstractmethod
    def cov(self, x: "Array", rowvar: bool, bias: bool) -> "Array":
        """Return the covariance matrix from a sample."""

    @abc.abstractmethod
    def det(self, x: "Array") -> "Array":
        """Return the determinant of a matrix."""

    @abc.abstractmethod
    def reshape(self, x: "Array", shape: int | tuple[int, ...]) -> "Array":
        """Reshape an array to a new ``shape``."""
