import importlib.metadata

__version__ = importlib.metadata.version("humanleague")

from _humanleague import SobolSequence, flatten, integerise, ipf, qis, qisi

from .ilp import ilp, ilp_ipf
from .utils import tabulate_counts, tabulate_individuals

__all__ = [
    "flatten",
    "integerise",
    "ilp",
    "ilp_ipf",
    "ipf",
    "qis",
    "qisi",
    "SobolSequence",
    "tabulate_counts",
    "tabulate_individuals",
]
