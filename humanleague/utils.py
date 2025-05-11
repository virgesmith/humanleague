from itertools import chain
import numpy as np
import numpy.typing as npt
import pandas as pd


def tabulate_counts(
    population: npt.NDArray, names: list[str] | tuple[str, ...] | None = None
) -> pd.Series:
    """
    Converts a multidimensional state population array into a pandas Series of counts of indexed by state.

    Parameters:
        population (npt.NDArray): A multidimensional NumPy array containing the data to be tabulated.
        names (list[str] | tuple[str, ...] | None, optional): A list or tuple of strings representing the names
            of the index levels for the resulting MultiIndex. If None, the index levels will be unnamed.

    Returns:
        pd.Series: A pandas Series where the index is a MultiIndex created from the shape of the input array,
        and the data corresponds to the flattened values of the input array.
    """
    index = pd.MultiIndex.from_tuples(list(np.ndindex(population.shape)), names=names)
    return pd.Series(
        index=index, data=list(np.nditer(population)), dtype=int, name="count"
    )


def tabulate_individuals(
    population: npt.NDArray, names: list[str] | tuple[str, ...] | None = None
) -> pd.DataFrame:
    """
    Converts a multidimensional population array into a tabular DataFrame format.

    This function takes a multidimensional array representing population counts
    and expands it into a DataFrame where each row corresponds to an individual,
    based on the counts in the input array. Optionally, column names can be provided
    for the resulting DataFrame.

    Args:
        population (npt.NDArray): A multidimensional NumPy array where each element
            represents the count of individuals in a specific category or combination
            of categories.
        names (list[str] | tuple[str, ...] | None, optional): A list or tuple of strings
            specifying the column names for the resulting DataFrame. If None, default
            column names will be used.

    Returns:
        pd.DataFrame: A pandas DataFrame where each row represents an individual,
        and the columns correspond to the indices of the input array.

    Example:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from your_module import tabulate_individuals
        >>> population = np.array([[2, 1], [0, 3]])
        >>> names = ["Category1", "Category2"]
        >>> df = tabulate_individuals(population, names)
        >>> print(df)
           Category1  Category2
        0          0          0
        1          0          0
        2          0          1
        3          1          1
        4          1          1
        5          1          1
    """

    exploded = chain.from_iterable(
        (idx,) * int(count)  # type: ignore[call-overload]
        for idx, count in zip(np.ndindex(population.shape), np.nditer(population))
    )

    return pd.DataFrame(data=exploded, columns=names)
