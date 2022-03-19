"""
    Microsynthesis using quasirandom sampling and IPF, plus related functionality
"""
from __future__ import annotations
import typing
import numpy as np
_Shape = typing.Tuple[int, ...]

__all__ = [
    "flatten",
    "integerise",
    "ipf",
    "prob2IntFreq",
    "qis",
    "qisi",
    "sobolSequence",
    "unittest"
]

T = typing.TypeVar("T")
nparray = np.ndarray[T, np.dtype[T]]

def flatten(pop: nparray[np.int64]) -> list:
    """
    Converts an n-dimensional array of counts into an n-column table with a row for each unit

    Args:

        pop: The population.

    Returns:

        A 2-d array of size n by sum(pop).
    """
@typing.overload
def integerise(frac: nparray[np.float64], pop: int) -> dict:
    """
    Computes the closest integer frequencies given fractional counts and a total population.

    Args:

        frac: The fractional counts (must be a 1-d array).

        pop: The total population

    Returns:

        A dictionary containing the frequencies and the RMS error


    Tries to construct and integer multidimensional array that has identical marginal sums to the fractional input array (which of course must have
    integer marginal sums). The algorithm may not always find a solution and will return an approximate array in this case.

    Args:

        pop: The fractional population.

    Returns:

        A dictionary containing The integral population, the RMS error, and a boolean indicating whether the population matches the marginal sums.
    """
@typing.overload
def integerise(pop: nparray[np.float64]) -> dict:
    pass
def ipf(seed: nparray[np.float64], indices: list[nparray[np.int64]], marginals: list[nparray[np.float64]]) -> dict:
    """
    Uses iterative proportional fitting to construct an n-dimensional array from a seed population that matches the specified marginal sums.

        seed: The seed population as an array.

        indices: A list of the indices in the overall array that each marginal represents

        marginals: A list of arrays containing the marginal sums.

    Returns:

        A dictionary containing the result, a convergence flag, the total population, the iterations and the error
    """
def prob2IntFreq(frac: nparray[np.float64], pop: int) -> dict:
    """
    Computes the closest integer frequencies given fractional counts and a total population.

    Args:

        frac: The fractional counts (must be a 1-d array).

        pop: The total population

    Returns:

        A dictionary containing the frequencies and the RMS error
    """
@typing.overload
def qis(indices: list[nparray[np.int64]], marginals: list[nparray[np.int64]]) -> dict:
    """
    Uses quasirandom integer sampling to construct an n-dimensional population array that matches the specified marginal sums.

        indices: A list of the indices in the overall array that each marginal represents

        marginals: A list of arrays containing the marginal sums.

        skips: The number of Sobol values to skip. NB the actual number skipped will be the largest power of 2 smaller than the supplied value.

    Returns:

        A dictionary containing the result, a convergence flag, the total population, the iterations and the some statistical measures.


    Uses quasirandom integer sampling to construct an n-dimensional population array that matches the specified marginal sums.

        indices: A list of the indices in the overall array that each marginal represents

        marginals: A list of arrays containing the marginal sums.

    Returns:

        A dictionary containing the result, a convergence flag, the total population, the iterations and the some statistical measures.
    """
@typing.overload
def qis(indices: list[nparray[np.int64]], marginals: list[nparray[np.int64]], skips: int) -> dict:
    pass
@typing.overload
def qisi(seed: nparray[np.float64], indices: list[nparray[np.int64]], marginals: list[nparray[np.int64]]) -> dict:
    """
    Uses quasirandom integer sampling to construct an n-dimensional population array that matches the specified marginal sums.

        seed: The dimension of the sequence (between 1 and 1111).

        indices: A list of the indices in the overall array that each marginal represents

        marginals: A list of arrays containing the marginal sums.

        skips: The number of Sobol values to skip. NB the actual number skipped will be the largest power of 2 smaller than the supplied value.

    Returns:

        A dictionary containing the result, a convergence flag, the total population, the iterations and the some statistical measures.


    Uses quasirandom integer sampling to construct an n-dimensional population array that matches the specified marginal sums.

        seed: The dimension of the sequence (between 1 and 1111).

        indices: A list of the indices in the overall array that each marginal represents

        marginals: A list of arrays containing the marginal sums.

    Returns:

        A dictionary containing the result, a convergence flag, the total population, the iterations and the some statistical measures.
    """
@typing.overload
def qisi(seed: nparray[np.float64], indices: list[nparray[np.int64]], marginals: list[nparray[np.int64]], skips: int) -> dict:
    pass
@typing.overload
def sobolSequence(dim: int, length: int) -> nparray[np.float64]:
    """
    Returns a Sobol' sequence given of supplied dimension and length, optionally skipping values.

        dim: The dimension of the sequence (between 1 and 1111).

        length: The length of the returned sequence

        skips: The number of values to skip. NB the actual number skipped will be the largest power of 2 smaller than the supplied value.

    Returns:

        A 2d array containing Sobol sequence values in (0,1).


    Returns a Sobol' sequence given of supplied dimension and length, optionally skipping values.

        dim: The dimension of the sequence (between 1 and 1111).

        length: The length of the returned sequence

    Returns:

        A 2d array containing Sobol sequence values in (0,1).
    """
@typing.overload
def sobolSequence(dim: int, length: int, skips: int) -> nparray[np.float64]:
    pass
def unittest() -> dict:
    """
    For developers. Runs the C++ unit tests.
    """
