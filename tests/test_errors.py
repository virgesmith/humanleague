import numpy as np
import pytest

import humanleague as hl


def test_errors() -> None:
    seed = np.ones(3)

    with pytest.raises(RuntimeError):
        hl.ipf(seed, [(0,)], [np.array([1, 2, 3])])
