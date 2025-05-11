import importlib.metadata

__version__ = importlib.metadata.version("humanleague")

from _humanleague import flatten, integerise, ipf, qis, qisi, SobolSequence
from .utils import tabulate_counts, tabulate_individuals

__all__ = [
    "flatten",
    "integerise",
    "ipf",
    "qis",
    "qisi",
    "SobolSequence",
    "tabulate_counts",
    "tabulate_individuals",
]
