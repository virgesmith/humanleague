#!/usr/bin/env python3

import glob

from pybind11.setup_helpers import ParallelCompile, Pybind11Extension
from setuptools import setup  # type: ignore


def source_files() -> list[str]:
    sources = glob.glob("src/*.cpp")
    # can't use compile skips as some files are auto-generated
    skip = ["src/RcppExports.cpp", "src/rcpp_api.cpp"]
    return [f for f in sources if f not in skip]


def header_files() -> list[str]:
    return glob.glob("src/*.h")


def defines() -> list[tuple[str, str | None]]:
    return [("PYTHON_MODULE", None)]


ext_modules = [
    Pybind11Extension(
        "_humanleague",
        sources=source_files(),
        include_dirs=["src"],
        define_macros=defines(),
        depends=["setup.py", "src/docstr.inl"] + header_files(),
        cxx_std=20,
    )
]


ParallelCompile().install()

setup(
    name="humanleague",
    packages=["humanleague"],
    package_data={"humanleague": ["py.typed", "*.pyi"]},
    ext_modules=ext_modules,
    zip_safe=False,
)
