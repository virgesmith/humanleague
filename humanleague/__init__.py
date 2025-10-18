import importlib.metadata

__version__ = importlib.metadata.version("humanleague")

from .humanleague_ext import SobolSequence, flatten, integerise, ipf, qis, qisi
from .utils import tabulate_counts, tabulate_individuals

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
