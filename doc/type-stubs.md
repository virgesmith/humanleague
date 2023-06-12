# updating type stubs

1. install `pybind11-stubgen`
1. run `pybind11-stubgen _humanleague`
1. copy `stubs/_humanleague/__init__.pyi` to `humanleague`
1. edit the file:
    1. delete the line `import _humanleague`
    1. import numpy.typing and edit numpy types as necessary. The following definitions are also useful:

        ```py
        FloatArray1d = npt.NDArray[np.float64] | list[float]
        IntArray1d = typing.Sequence[int]
        ```

    1. move misplaced docstrs for overloaded functions/methods as necessary
    1. replace `__version__ = ...` with `__version__: str`