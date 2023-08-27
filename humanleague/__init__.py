import importlib.metadata

__version__ = importlib.metadata.version("humanleague")

from _humanleague import flatten, integerise, ipf, qis, qisi, SobolSequence
