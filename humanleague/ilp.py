from math import prod
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
from itrx import Itr
from scipy.optimize import Bounds, LinearConstraint, milp

# 1. ILP as a replacement for QIS (no seed) - inputs are marginals.
# 1a. can choice of seed
# 2. ILP as a replacement for multidim integerisation - input is fractional population matrix


def _strides(m: npt.NDArray) -> tuple[int, ...]:
    return (1, *m.shape[:-1])


def _sum_over(result: npt.NDArray[np.int64], indices: Sequence[int]) -> npt.NDArray[np.int64]:
    sum_dims = tuple(d for d in range(result.ndim) if d not in indices)
    print(sum_dims)
    # empty tuple
    return result.sum(axis=sum_dims)


def ilp(
    indices: Sequence[Sequence[int] | int],
    marginals: Sequence[npt.NDArray[np.int64]],
    *,
    lbound: npt.NDArray[np.float64] | None = None,
    ubound: npt.NDArray[np.float64] | None = None,
) -> tuple[npt.NDArray[np.int64], dict[str, Any]]:
    indices = Itr(indices).map(lambda idx: (idx,) if isinstance(idx, int) else idx).collect()
    n_dim = len(Itr(indices).flatten().collect(set))

    # determine shape of output
    shapelist = [0] * n_dim
    for index, marginal in zip(indices, marginals, strict=True):
        for i, idx in enumerate(index):
            if shapelist[idx] > 0 and shapelist[idx] != marginal.shape[i]:
                raise ValueError("Inconsistent marginal dimensions")
            shapelist[idx] = marginal.shape[i]
    shape = tuple(shapelist)

    if lbound is not None and lbound.shape != shape:
        raise ValueError("lbound dimensions are inconsistent with marginals", lbound.shape, shape)
    if ubound is not None and ubound.shape != shape:
        raise ValueError("ubound dimensions are inconsistent with marginals", ubound.shape, shape)

    # determine total population
    pop = marginals[0].sum()
    for m in marginals[1:]:
        if m.sum() != pop:
            raise ValueError("Inconsistent marginal sums")

    # check bounds
    if lbound is not None and lbound.sum() > pop:
        raise ValueError("Lower bound is too large")

    if ubound is not None and ubound.sum() < pop:
        raise ValueError("Upper bound is too small")

    if lbound is not None and ubound is not None:
        if (ubound - lbound < 0).any():
            raise ValueError("Bounds are inconsistent (upper<lower)")

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

    lbound = np.full(n_states, 0) if lbound is None else lbound.reshape(n_states)
    ubound = np.full(n_states, pop) if ubound is None else ubound.reshape(n_states)
    bounds = Bounds(lb=lbound, ub=ubound)

    x0 = np.ones(n_states)

    res = milp(x0, bounds=bounds, constraints=constraints, integrality=integrality)

    if res.x is None:
        raise ValueError("milp did not return a result")

    solution = res.x.reshape(shape).astype(int)

    conv = res.success
    for idx, marginal in zip(indices, marginals, strict=True):
        conv &= (_sum_over(solution, idx) == marginal).all()

    return solution, {"conv": conv, "pop": int(res.x.sum())}
