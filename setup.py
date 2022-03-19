#!/usr/bin/env python3

import glob
from setuptools import setup  # type: ignore
from pybind11.setup_helpers import Pybind11Extension, ParallelCompile


def readme():
  with open('README.md') as f:
    return f.read()


def version():
  """ The R file DESCRIPTION in the project root is now the single source of version info """
  with open("DESCRIPTION") as fd:
    lines = fd.readlines()
    for line in lines:
      if line.startswith("Version:"):
        return line.rstrip().split(":")[1].lstrip()


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
    ("HUMANLEAGUE_VERSION", version()),
    ("PYTHON_MODULE", None)
  ]


ext_modules = [
  Pybind11Extension(
    '_humanleague',
    sources=source_files(),
    include_dirs=["src"],
    define_macros=defines(),
    depends=["setup.py", "DESCRIPTION", "src/docstr.inl"] + header_files(),
    cxx_std=17
  )
]


ParallelCompile().install()

setup(
  name='humanleague',
  version=version(),
  description='Microsynthesis using quasirandom sampling and/or IPF',
  author='Andrew P Smith',
  author_email='a.p.smith@leeds.ac.uk',
  url='http://github.com/virgesmith/humanleague',
  long_description=readme(),
  long_description_content_type="text/markdown",
  packages=["humanleague"],
  package_data={"humanleague": ["py.typed", "*.pyi"]}, ext_modules=ext_modules,
  install_requires=['numpy>=1.19.1'],
  setup_requires=['pybind11>=2.5.0', 'pytest-runner'],
  tests_require=['pytest'],
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ],
  zip_safe=False,
)
