"""
Microsynthesis using quasirandom sampling and IPF, plus related functionality
"""

from __future__ import annotations
import typing
import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray1d = npt.NDArray[np.float64] | list[float]
IntArray1d = typing.Sequence[int]

__all__ = [
    "SobolSequence",
    "flatten",
    "integerise",
    "ipf",
    "qis",
    "qisi",
    "tabulate_counts",
    "tabulate_individuals",
]

def tabulate_counts(
    population: npt.NDArray, names: list[str] | tuple[str, ...] | None = None
) -> pd.Series: ...
def tabulate_individuals(
    population: npt.NDArray, names: list[str] | tuple[str, ...] | None = None
) -> pd.DataFrame: ...

class SobolSequence:
    @typing.overload
    def __init__(self, dim: int) -> None:
        """
        Construct a `dim` dimensional Sobol sequence generator object.

        Args:

            dim: The dimension of the sequence (between 1 and 1111).

        Returns:

            A generator object that produces Sobol sequence values in (0,1)^dim.
        """
    @typing.overload
    def __init__(self, dim: int, skips: int) -> None:
        """
        Construct a `dim` dimensional Sobol sequence generator object, skipping the start of the sequence.

        Args:

            dim: The dimension of the sequence (between 1 and 1111).

            skips: The number of values to skip. NB the actual number skipped will be the largest power of 2 smaller than the supplied value.

        Returns:

            A generator object that produces Sobol sequence values in (0,1)^dim.
        """
    def __iter__(self) -> SobolSequence:
        """
        __iter__ dunder
        """
    def __next__(self) -> npt.NDArray[np.float64]:
        """
        __next__ dunder
        """
    pass

def _unittest() -> dict:
    """
    For developers. Runs the C++ unit tests.
    """

def flatten(pop: npt.NDArray[np.int64]) -> list:
    """
    Converts an n-dimensional array of counts into an n-column table with a row for each unit

    Args:

        pop: The population.

    Returns:

        A 2-d array of size n by sum(pop).
    """

@typing.overload
def integerise(
    frac: FloatArray1d, pop: int
) -> tuple[npt.NDArray[np.int64], dict[str, typing.Any]]:
    """
    Computes the closest integer frequencies given fractional counts and a total population.

    Args:

        frac: The fractional counts (must be a 1-d array).

        pop: The total population

    Returns:

        A tuple containing the result and summary statistics
    """

@typing.overload
def integerise(
    pop: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.int64], dict[str, typing.Any]]:
    """
    Tries to construct an integer multidimensional array that has identical marginal sums to the fractional input array (which of course must have
    integer marginal sums). The algorithm may not always find a solution and will return an approximate array in this case.

    Args:

        pop: The fractional population.

    Returns:

        A tuple containing the result and summary statistics
    """

def ipf(
    seed: npt.NDArray[np.float64],
    indices: typing.Iterable[IntArray1d],
    marginals: typing.Iterable[npt.NDArray[np.float64]],
) -> tuple[npt.NDArray[np.float64], dict[str, typing.Any]]:
    """
    Uses iterative proportional fitting to construct an n-dimensional array from a seed population that matches the specified marginal sums.

        seed: The seed population or distribution.

        indices: A list of the indices in the overall array that each marginal represents

        marginals: A list of arrays containing the marginal sums.

    Returns:

        A tuple containing the result and summary statistics
    """

@typing.overload
def qis(
    indices: typing.Iterable[IntArray1d],
    marginals: typing.Iterable[npt.NDArray[np.int64]],
) -> tuple[npt.NDArray[np.int64], dict[str, typing.Any]]:
    """
    Uses quasirandom integer sampling to construct an n-dimensional population array that matches the specified marginal sums.

        indices: A list of the indices in the overall array that each marginal represents

        marginals: A list of arrays containing the marginal sums.

    Returns:

        A tuple containing the result and summary statistics
    """

@typing.overload
def qis(
    indices: typing.Iterable[IntArray1d],
    marginals: typing.Iterable[npt.NDArray[np.int64]],
    skips: int,
) -> tuple[npt.NDArray[np.int64], dict[str, typing.Any]]:
    """
    Uses quasirandom integer sampling to construct an n-dimensional population array that matches the specified marginal sums.

        indices: A list of the indices in the overall array that each marginal represents

        marginals: A list of arrays containing the marginal sums.

        skips: The number of Sobol values to skip. NB the actual number skipped will be the largest power of 2 smaller than the supplied value.

    Returns:

        A tuple containing the result and summary statistics
    """

@typing.overload
def qisi(
    seed: npt.NDArray[np.float64],
    indices: typing.Iterable[IntArray1d],
    marginals: typing.Iterable[npt.NDArray[np.int64]],
) -> tuple[npt.NDArray[np.int64], dict[str, typing.Any]]:
    """
    Uses quasirandom integer sampling to construct an n-dimensional population array that matches the specified marginal sums.

        seed: The seed population or distribution.

        indices: A list of the indices in the overall array that each marginal represents

        marginals: A list of arrays containing the marginal sums.

    Returns:

        A tuple containing the result and summary statistics
    """

@typing.overload
def qisi(
    seed: npt.NDArray[np.float64],
    indices: list[IntArray1d],
    marginals: list[npt.NDArray[np.int64]],
    skips: int,
) -> tuple[npt.NDArray[np.int64], dict[str, typing.Any]]:
    """
    Uses quasirandom integer sampling to construct an n-dimensional population array that matches the specified marginal sums.

        seed: The seed population or distribution.

        indices: A list of the indices in the overall array that each marginal represents

        marginals: A list of arrays containing the marginal sums.

        skips: The number of Sobol values to skip. NB the actual number skipped will be the largest power of 2 smaller than the supplied value.

    Returns:

        A tuple containing the result and summary statistics
    """

__version__: str
