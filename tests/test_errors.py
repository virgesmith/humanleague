import numpy as np
import pytest

import humanleague as hl


def test_input_validation() -> None:

    m1 = np.array([1, 2, 3])
    m2 = np.ones((2,3))

    with pytest.raises(RuntimeError):
        # index list size 1 too small or differs from marginal size 1
        hl.ipf(np.ones(3), [(0,)], [m1])

    with pytest.raises(RuntimeError):
        # problem needs to have more than 1 dimension!
        hl.ipf(np.ones(3),[(0,), (0,)], [m1, m1])

    with pytest.raises(RuntimeError):
        # index/marginal dimension mismatch 2 vs 1
        hl.ipf(np.ones(3),[(0,), (1, 2)], [m1, m1])

    with pytest.raises(RuntimeError):
        #  mismatch at index 1: dimension 1 size 3 redefined to 2
        hl.ipf(np.ones((3, 3)), [(0, 1), (1, 0)], [m2, m2])

    with pytest.raises(RuntimeError):
        #  negative value in marginal 0: -3.000000
        hl.ipf(np.ones((3, 3)), [(0,), (1,)], [m1, -m1])
