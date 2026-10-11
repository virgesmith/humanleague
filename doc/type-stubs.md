# updating type stubs

`humanleague/__init__.pyi` is hand-maintained. `nanobind.stubgen` is used to detect changes to the bound signatures, but
its output is less precise than the hand-written stubs (bare `tuple` return types, untyped `Iterable` arguments) and
omits everything defined in Python (`tabulate_counts`, `tabulate_individuals`, `__version__`). So never write its output
over the stubs directly.

1. generate the stubs to a temporary file:

    ```sh
    uv run python -m nanobind.stubgen -P -m humanleague.humanleague_ext -o /tmp/humanleague_ext.pyi
    ```

1. diff against the current stubs and copy across only genuine signature changes (new/removed functions or overloads,
   renamed or added parameters, changed docstrings):

    ```sh
    diff humanleague/__init__.pyi /tmp/humanleague_ext.pyi
    ```

1. when copying, keep the hand-written conventions:
    1. precise numpy types via `numpy.typing`, using these aliases where they apply:

        ```py
        FloatArray1d = npt.NDArray[np.float64] | list[float]
        IntArray1d = Sequence[int]
        ```

    1. explicit return types, e.g. `tuple[npt.NDArray[np.int64], dict[str, Any]]` rather than `tuple`
    1. `collections.abc` / `typing` imports by name (`Iterable`, `overload`), as stubgen writes them
    1. `__version__: str` and the `tabulate_*` functions from `utils.py`

1. run `uv run ty check` and `uv run pytest` — the tests are type-checked against the stubs.
