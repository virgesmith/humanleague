# v2.5.0

- switch the python build system and bindings from `setuptools`/`pybind11` to `scikit-build-core`/`nanobind` (CMake-based), targeting the CPython 3.12+ stable ABI.
  - substantially faster local rebuilds and a smaller compiled extension.
  - no runtime or API behaviour changes are intended: `ipf`, `qis` and `qisi` still accept scalars, numpy integer scalars, lists, tuples or arrays for index elements.
  - the compiled extension module is now importable as `humanleague.humanleague_ext` (was `_humanleague`); this is an internal implementation detail and does not affect the public `humanleague` API.
  - type stubs are now regenerated with `nanobind.stubgen` instead of `pybind11-stubgen`, see [doc/type-stubs.md](./doc/type-stubs.md).

# v2.4.0

- introduces the `tabulate_counts` and `tabulate_individuals` functions.
  - `tabulate_counts`: converts a multidimensional integer array to a pandas Series of state counts, indexed by the original array index
  - `tabulate_individuals`: converts a multidimensional integer state count array to a pandas DataFrame, with each row corresponding to an individual in the population.

- deprecates `flatten`. The `tabulate_individuals` function provides similar (but improved) functionality. To replicate the original behaviour use:
    ```py
    import humanleague as hl
    ...
    # p is an n-d array produced by (e.g.) QISI
    # equivalent to hl.flatten(p):
    flatten_result = hl.tabulate_individuals(p).to_numpy().T.tolist()
    ```

- The indices and marginals inputs to the `ipf`, `qis` and `qisi` functions are now iterables. This effectively means they can now be `tuple` or even `np.array` types, where previously they had to be `list`.
- for 1-dimensional marginals, indices can now be represented as a scalar, where previously a length-1 array (or tuple) was required:
    ```py
    # previously:
    result, stats = hl.qis([(0,), (1,)], [m0, m1])
    # now simpler:
    result, stats = hl.qis((0, 1), (m0, m1))
    # or even
    result, stats = hl.qis(range(3), (m0, m1, m2))
    ```

# v2.4.1

- fixes a bug in IPF where fractional marginals were inadvertently rounded down, producing incorrect results. This bug
was introduced in 2.4.0 and only affected the python version.

# v2.4.2

- fixes a bug where for integerising 1d arrays without explicitly providing an integer total, sometimes the computed
total was rounded down, resulting in 1 unit missing from the population.
- for consistency, add the "conv" key to the `stats` output for the 1d integerisation with a specified total case

# v2.4.3

- update supported python versions to 3.12, 3.13, 3.14
- fix issue with compiler error on debian bookworm
- linting

# v2.4.4

- improve input validation for `integerise`
- internal refactoring:
    - switch type checker from `mypy` to `ty`, update CI
    - stricter linting
    - add pre-commit hooks
    - format C++ sources