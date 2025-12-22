from math import prod
from typing import Any, Iterable

import numpy as np
import numpy.typing as npt
from itrx import Itr
from scipy.optimize import LinearConstraint, milp

import humanleague as hl

# 1. ILP as a replacement for QIS (no seed) - inputs are marginals.
# 1a. can choice of seed
# 2. ILP as a replacement for multidim integerisation - input is fractional population matrix

def _strides(m: npt.NDArray) -> tuple[int, ...]:
    return (1, *m.shape[:-1])


def ilp(indices: Iterable[Iterable[int]], marginals: list[npt.NDArray]) -> tuple[npt.NDArray[np.int64], dict[str, Any]]:
    indices = Itr(indices).map(lambda idx: (idx,) if isinstance(idx, int) else idx).collect()
    n_dim = len(Itr(indices).flatten().collect(set))

    # determine shape of output
    shape = [0] * n_dim
    for index, marginal in zip(indices, marginals, strict=True):
        for i, idx in enumerate(index):
            if shape[idx] > 0 and shape[idx] != marginal.shape[i]:
                raise ValueError("Inconsistent marginal dimensions")
            shape[idx] = marginal.shape[i]
    shape = tuple(shape)

    # determine total population
    pop = marginals[0].sum()
    for m in marginals[1:]:
        if m.sum() != pop:
            raise ValueError("Inconsistent marginal sums")

    n_states = prod(shape)

    # create matrices from flattened marginals
    A = [np.zeros((prod(m.shape), n_states), dtype=int) for m in marginals]

    for i, ix in enumerate(np.ndindex(shape)):
        for j in range(len(marginals)):
            # print(indices[j], marginals[j].strides)
            n = sum(ix[k] * s for k, s in zip(indices[j], _strides(marginals[j]), strict=True))
            # print(A[j].shape, j, ix, (n, i))
            A[j][n, i] = 1
    assert all(Ai.sum() == n_states for Ai in A)

    # TODO perhaps a transpose is needed before flattening? yes
    constraints = [LinearConstraint(Ai, mi.T.flatten()) for Ai, mi in zip(A, marginals, strict=True)]
    integrality = np.ones(n_states, dtype=int)
    # TODO seed?
    x0 = np.full(n_states, marginals[0].sum() / n_states)

    res = milp(x0, constraints=constraints, integrality=integrality)

    # assert res.success

    return res.x.reshape(shape).astype(int), {"conv": res.success, "pop": int(res.x.sum())}

