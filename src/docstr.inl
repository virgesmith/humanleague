
const char* module_docstr = R"docstr(
    Microsynthesis using quasirandom sampling and IPF, plus related functionality
)docstr";

const char* version_docstr = R"docstr(
    Gets the module version
)docstr";

const char* prob2IntFreq_docstr = R"docstr(
    Computes the closest integer frequencies given fractional counts and a total population. 
    Args:
        probs: The probabilities.
        pop: The growth rate
    Returns:
        The frequencies and the RMS error
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
    Returns a Sobol' sequence given of supplied dimension and length, optionally skipping values:
        dim: The dimension of the sequence (between 1 and 1111).
        length: The length of the returned sequence
        skips: The number of values to skip. NB the actual number skipped will be the largest power of 2 smaller than the supplied value.
    Returns:
        A 2d array containing Sobol sequence values in (0,1). 
)docstr";

