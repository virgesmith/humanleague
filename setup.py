#!/usr/bin/env python3

from pathlib import Path

from pybind11.setup_helpers import ParallelCompile, Pybind11Extension
from setuptools import setup  # type: ignore


def _source_files() -> list[Path]:
    sources = Path("src").glob("*.cpp")
    # can't use compile skips as some files are auto-generated
    skip = ["RcppExports.cpp", "rcpp_api.cpp"]
    return [file for file in sources if not any(s in str(file) for s in skip)]


def _header_files() -> list[Path]:
    return list(Path("src").glob("*.h"))


def _defines() -> list[tuple[str, str | None]]:
    return [("PYTHON_MODULE", None)]


ext_modules = [
    Pybind11Extension(
        "_humanleague",
        sources=_source_files(),
        include_dirs=["src"],
        define_macros=_defines(),
        depends=["setup.py", "src/docstr.inl", *_header_files()],
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
