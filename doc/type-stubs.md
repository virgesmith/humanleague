# updating type stubs

1. install `pybind11-stubgen`
1. run `pybind11-stubgen _humanleague`
1. copy `stubs/_humanleague/__init__.pyi` to `humanleague`
1. edit the file:
    1. delete the line `import _humanleague`
    1. add a type alias for numpy arrays:
        ```py
        T = typing.TypeVar("T")
        nparray = numpy.ndarray[T, numpy.dtype[T]]
        ```
    1. move docstrs for overloaded functions/methods as necessary