# updating type stubs

1. run

    ```sh
    uv run python -m nanobind.stubgen -P -m humanleague.humanleague_ext -o humanleague/__init__.pyi -M humanleague/py.typed
    ```

1. edit the file:
    1. delete the line `import humanleague.humanleague_ext`
    1. import numpy.typing and edit numpy types as necessary. The following definitions are also useful:

        ```py
        FloatArray1d = npt.NDArray[np.float64] | list[float]
        IntArray1d = typing.Sequence[int]
        ```

    1. move misplaced docstrs for overloaded functions/methods as necessary
    1. replace `__version__ = ...` with `__version__: str`