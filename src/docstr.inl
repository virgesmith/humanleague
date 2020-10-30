
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