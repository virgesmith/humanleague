from math import prod
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt
from itrx import Itr
from scipy.optimize import Bounds, LinearConstraint, milp

from humanleague import ipf

# 1. ILP as a replacement for QIS (no seed) - inputs are marginals.
# 1a. can choice of seed
# 2. ILP as a replacement for multidim integerisation - input is fractional population matrix


def _strides(m: npt.NDArray) -> tuple[int, ...]:
    return (1, *m.shape[:-1])


def _sum_over(result: npt.NDArray[np.int64], indices: Sequence[int]) -> npt.NDArray[np.int64]:
    sum_dims = tuple(d for d in range(result.ndim) if d not in indices)
    return result.sum(axis=sum_dims)


def _check_bounds(
    lbound: npt.NDArray[np.float64] | None, ubound: npt.NDArray[np.float64] | None, shape: tuple[int, ...], pop: int
) -> None:
    if lbound is not None and lbound.shape != shape:
        raise ValueError("lbound dimensions are inconsistent with marginals", lbound.shape, shape)
    if ubound is not None and ubound.shape != shape:
        raise ValueError("ubound dimensions are inconsistent with marginals", ubound.shape, shape)

    if lbound is not None and lbound.sum() > pop:
        raise ValueError("Lower bound is too large")

    if ubound is not None and ubound.sum() < pop:
        raise ValueError("Upper bound is too small")

    if lbound is not None and ubound is not None:
        if (ubound - lbound < 0).any():
            raise ValueError("Bounds are inconsistent (upper<lower)")


# determine shape of output
def _get_shape(indices, marginals) -> tuple[int, ...]:
    n_dim = len(Itr(indices).flatten().collect(set))
    shapelist = [0] * n_dim
    for index, marginal in zip(indices, marginals, strict=True):
        for i, idx in enumerate(index):
            if shapelist[idx] > 0 and shapelist[idx] != marginal.shape[i]:
                raise ValueError("Inconsistent marginal dimensions")
            shapelist[idx] = marginal.shape[i]
    return tuple(shapelist)


# determine total population and check marginals are consistent
def _get_population(marginals: Sequence[npt.NDArray[np.int64]]) -> int:
    pop = Itr(marginals).map(np.ndarray.sum).collect(set)
    if len(pop) > 1:
        raise ValueError("Inconsistent marginal sums")
    return pop.pop()


def ilp(
    indices: Sequence[Sequence[int] | int],
    marginals: Sequence[npt.NDArray[np.int64]],
    *,
    lbound: npt.NDArray[np.float64] | None = None,
    ubound: npt.NDArray[np.float64] | None = None,
) -> tuple[npt.NDArray[np.int64], dict[str, Any]]:
    indices = Itr(indices).map(lambda idx: (idx,) if isinstance(idx, int) else idx).collect()

    shape = _get_shape(indices, marginals)

    pop = _get_population(marginals)

    _check_bounds(lbound, ubound, shape, pop)

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
    for ix, marginal in zip(indices, marginals, strict=True):  # type: ignore[assignment]
        conv &= (_sum_over(solution, ix) == marginal).all()

    return solution, {"conv": conv, "pop": int(res.x.sum())}


def ilp_ipf(
    seed: npt.NDArray[np.float64],
    indices: Sequence[Sequence[int] | int],
    marginals: Sequence[npt.NDArray[np.int64]],
) -> tuple[npt.NDArray[np.int64], dict[str, Any]]:
    """Use IPF, then bounded ILP with widening integer bounds"""
    indices = Itr(indices).map(lambda idx: (idx,) if isinstance(idx, int) else idx).collect()

    shape = _get_shape(indices, marginals)

    # determine total population
    _pop = _get_population(marginals)

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

    ipf_solution, ipf_stats = ipf(seed, indices, marginals)  # type: ignore[arg-type]

    if not ipf_stats["conv"]:
        raise ValueError("IPF failed")

    x0 = np.ones(n_states)

    # widen bounds until a solution is found
    width = 0
    while width < 4:
        lbound = np.clip(np.rint(ipf_solution - width), 0, None).reshape(n_states)
        ubound = np.rint(ipf_solution + 1 + width).reshape(n_states)

        bounds = Bounds(lb=lbound, ub=ubound)

        res = milp(x0, bounds=bounds, constraints=constraints, integrality=integrality)

        if res.x is None:
            continue

        solution = res.x.reshape(shape).astype(int)

        conv = res.success
        for ix, marginal in zip(indices, marginals, strict=True):  # type: ignore[assignment]
            conv &= (_sum_over(solution, ix) == marginal).all()
        if conv:
            return solution, {"conv": conv, "pop": int(res.x.sum())}
        width += 1

    raise ValueError("milp integerisation failure")
