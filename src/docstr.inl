
const char* module_docstr = R"""(
    Microsynthesis using quasirandom sampling and IPF, plus related functionality
)""";

const char* flatten_docstr = R"""(
    Converts an n-dimensional array of counts into an n-column table with a row for each unit

    Args:

        pop: The population.

    Returns:

        A 2-d array of size n by sum(pop).
)""";

const char* integerise1d_docstr = R"""(
    Computes the closest integer frequencies given fractional counts and a total population.

    Args:

        frac: The fractional counts (must be a 1-d array).

        pop: The total population

    Returns:

        A dictionary containing the frequencies and the RMS error
)""";

const char* integerise_docstr = R"""(
    Tries to construct an integer multidimensional array that has identical marginal sums to the fractional input array (which of course must have
    integer marginal sums). The algorithm may not always find a solution and will return an approximate array in this case.

    Args:

        pop: The fractional population.

    Returns:

        A dictionary containing The integral population, the RMS error, and a boolean indicating whether the population matches the marginal sums.
)""";

const char* SobolSequence_docstr = R"""(
    Generator that returns the next point in a Sobol' sequence given of supplied dimension, optionally skipping values.

        dim: The dimension of the sequence (between 1 and 1111).

        skips: The number of values to skip. NB the actual number skipped will be the largest power of 2 smaller than the supplied value.

    Returns:

        A generator object that produces Sobol sequence values in (0,1).
)""";

const char* ipf_docstr = R"""(
    Uses iterative proportional fitting to construct an n-dimensional array from a seed population that matches the specified marginal sums.

        seed: The seed population or distribution.

        indices: A list of the indices in the overall array that each marginal represents

        marginals: A list of arrays containing the marginal sums.

    Returns:

        A dictionary containing the result, a convergence flag, the total population, the iterations and the error
)""";

const char* qis_docstr = R"""(
    Uses quasirandom integer sampling to construct an n-dimensional population array that matches the specified marginal sums.

        indices: A list of the indices in the overall array that each marginal represents

        marginals: A list of arrays containing the marginal sums.

        skips: The number of Sobol values to skip. NB the actual number skipped will be the largest power of 2 smaller than the supplied value.

    Returns:

        A dictionary containing the result, a convergence flag, the total population, the iterations and the some statistical measures.
)""";

const char* qis2_docstr = R"""(
    Uses quasirandom integer sampling to construct an n-dimensional population array that matches the specified marginal sums.

        indices: A list of the indices in the overall array that each marginal represents

        marginals: A list of arrays containing the marginal sums.

    Returns:

        A dictionary containing the result, a convergence flag, the total population, the iterations and the some statistical measures.
)""";

const char* qisi_docstr = R"""(
    Uses quasirandom integer sampling to construct an n-dimensional population array that matches the specified marginal sums.

        seed: The seed population or distribution.

        indices: A list of the indices in the overall array that each marginal represents

        marginals: A list of arrays containing the marginal sums.

        skips: The number of Sobol values to skip. NB the actual number skipped will be the largest power of 2 smaller than the supplied value.

    Returns:

        A dictionary containing the result, a convergence flag, the total population, the iterations and the some statistical measures.
)""";

const char* qisi2_docstr = R"""(
    Uses quasirandom integer sampling to construct an n-dimensional population array that matches the specified marginal sums.

        seed: The seed population or distribution.

        indices: A list of the indices in the overall array that each marginal represents

        marginals: A list of arrays containing the marginal sums.

    Returns:

        A dictionary containing the result, a convergence flag, the total population, the iterations and the some statistical measures.
)""";

const char* unittest_docstr = R"""(
    For developers. Runs the C++ unit tests.
)""";

