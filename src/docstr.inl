
const char* module_docstr = R"docstr(
    Microsynthesis using quasirandom sampling and IPF, plus related functionality
)docstr";

const char* flatten_docstr = R"docstr(
    Converts an n-dimensional array of counts into an n-column table with a row for each unit

    Args:

        pop: The population.

    Returns:

        A 2-d array of size n by sum(pop).
)docstr";

const char* prob2IntFreq_docstr = R"docstr(
    Computes the closest integer frequencies given fractional counts and a total population.

    Args:

        frac: The fractional counts (must be a 1-d array).

        pop: The total population

    Returns:

        A dictionary containing the frequencies and the RMS error
)docstr";

const char* integerise_docstr = R"docstr(
    Tries to construct and integer multidimensional array that has identical marginal sums to the fractional input array (which of course must have
    integer marginal sums). The algorithm may not always find a solution and will return an approximate array in this case.

    Args:

        pop: The fractional population.

    Returns:

        A dictionary containing The integral population, the RMS error, and a boolean indicating whether the population matches the marginal sums.
)docstr";


const char* sobolSequence_docstr = R"docstr(
    Returns a Sobol' sequence given of supplied dimension and length, optionally skipping values.

        dim: The dimension of the sequence (between 1 and 1111).

        length: The length of the returned sequence

        skips: The number of values to skip. NB the actual number skipped will be the largest power of 2 smaller than the supplied value.

    Returns:

        A 2d array containing Sobol sequence values in (0,1).
)docstr";

const char* sobolSequence2_docstr = R"docstr(
    Returns a Sobol' sequence given of supplied dimension and length, optionally skipping values.

        dim: The dimension of the sequence (between 1 and 1111).

        length: The length of the returned sequence

    Returns:

        A 2d array containing Sobol sequence values in (0,1).
)docstr";

const char* SobolSequence_docstr = R"docstr(
    Generator that returns the next value in a Sobol' sequence given of supplied dimension, optionally skipping values.

        dim: The dimension of the sequence (between 1 and 1111).

        length: The length of the returned sequence

        skips: The number of values to skip. NB the actual number skipped will be the largest power of 2 smaller than the supplied value.

    Returns:

        A generator object that produces Sobol sequence values in (0,1).
)docstr";


const char* ipf_docstr = R"docstr(
    Uses iterative proportional fitting to construct an n-dimensional array from a seed population that matches the specified marginal sums.

        seed: The seed population or distribution.

        indices: A list of the indices in the overall array that each marginal represents

        marginals: A list of arrays containing the marginal sums.

    Returns:

        A dictionary containing the result, a convergence flag, the total population, the iterations and the error
)docstr";

const char* qis_docstr = R"docstr(
    Uses quasirandom integer sampling to construct an n-dimensional population array that matches the specified marginal sums.

        indices: A list of the indices in the overall array that each marginal represents

        marginals: A list of arrays containing the marginal sums.

        skips: The number of Sobol values to skip. NB the actual number skipped will be the largest power of 2 smaller than the supplied value.

    Returns:

        A dictionary containing the result, a convergence flag, the total population, the iterations and the some statistical measures.
)docstr";

const char* qis2_docstr = R"docstr(
    Uses quasirandom integer sampling to construct an n-dimensional population array that matches the specified marginal sums.

        indices: A list of the indices in the overall array that each marginal represents

        marginals: A list of arrays containing the marginal sums.

    Returns:

        A dictionary containing the result, a convergence flag, the total population, the iterations and the some statistical measures.
)docstr";

const char* qisi_docstr = R"docstr(
    Uses quasirandom integer sampling to construct an n-dimensional population array that matches the specified marginal sums.

        seed: The seed population or distribution.

        indices: A list of the indices in the overall array that each marginal represents

        marginals: A list of arrays containing the marginal sums.

        skips: The number of Sobol values to skip. NB the actual number skipped will be the largest power of 2 smaller than the supplied value.

    Returns:

        A dictionary containing the result, a convergence flag, the total population, the iterations and the some statistical measures.
)docstr";

const char* qisi2_docstr = R"docstr(
    Uses quasirandom integer sampling to construct an n-dimensional population array that matches the specified marginal sums.

        seed: The seed population or distribution.

        indices: A list of the indices in the overall array that each marginal represents

        marginals: A list of arrays containing the marginal sums.

    Returns:

        A dictionary containing the result, a convergence flag, the total population, the iterations and the some statistical measures.
)docstr";

const char* unittest_docstr = R"docstr(
    For developers. Runs the C++ unit tests.
)docstr";

