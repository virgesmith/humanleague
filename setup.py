#!/usr/bin/env python3

import glob
from setuptools import setup  # type: ignore
from pybind11.setup_helpers import Pybind11Extension, ParallelCompile


def source_files():
  sources = glob.glob("src/*.cpp")
  # can't use compile skips as some files are auto-generated
  skip = ["RcppExports.cpp", "rcpp_api.cpp"]
  for s in skip:
    sources = [f for f in sources if s not in f]

  return sources


def header_files():
  return glob.glob("src/*.h")


def defines():
  return [
    ("PYTHON_MODULE", None)
  ]


ext_modules = [
  Pybind11Extension(
    '_humanleague',
    sources=source_files(),
    include_dirs=["src"],
    define_macros=defines(),
    depends=["setup.py", "src/docstr.inl"] + header_files(),
    cxx_std=20,
  )
]


ParallelCompile().install()

setup(
  name='humanleague',
  packages=["humanleague"],
  package_data={"humanleague": ["py.typed", "*.pyi"]},
  ext_modules=ext_modules,
  zip_safe=False,
)
