#!/usr/bin/env python3

import os
import glob
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools

# see https://github.com/pybind/python_example

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
  return [
    "src/Index.cpp",
    "src/Integerise.cpp",
    "src/module.cpp",
    "src/NDArrayUtils.cpp",
    "src/QIS.cpp",
    "src/QISI.cpp",
    "src/Sobol.cpp",
    "src/SobolImpl.cpp",
    "src/StatFuncs.cpp",
    "src/TestIndex.cpp",
    "src/TestNDArray.cpp",
    "src/TestReduce.cpp",
    "src/TestSlice.cpp",
    "src/TestSobol.cpp",
    "src/TestStatFuncs.cpp",
    "src/UnitTester.cpp"
  ]

def header_files():
  return [
    "src/DDWR.h",
    "src/Global.h",
    "src/Index.h",
    "src/Integerise.h",
    "src/IPF.h",
    "src/Log.h",
    "src/Microsynthesis.h",
    "src/NDArray.h",
    "src/NDArrayUtils.h",
    "src/QIS.h",
    "src/QISI.h",
    "src/SobolData.h",
    "src/Sobol.h",
    "src/SobolImpl.h",
    "src/StatFuncs.h",
    "src/UnitTester.h"   
  ]


def cxxflags(platform):
  if platform == "unix":
    return [
      "-Wall",
      "-pedantic",
      "-pthread",
      "-Wsign-compare",
      "-fstack-protector-strong",
      "-Wformat",
      "-Werror=format-security",
      "-Wdate-time",
      "-fPIC",
      "-std=c++11", # Rcpp compatibility
      "-fvisibility=hidden"
    ]
  elif platform == "msvc":
    return ['/EHsc']
  else:
    return []

def ldflags(_platform):
  return []

def defines(platform):
  return [
    ("HUMANLEAGUE_VERSION", version()),
    ("PYTHON_MODULE", None)
  ]

class get_pybind_include(object):
  """Helper class to determine the pybind11 include path

  The purpose of this class is to postpone importing pybind11
  until it is actually installed, so that the ``get_include()``
  method can be invoked. """

  def __str__(self):
    import pybind11
    return pybind11.get_include()


ext_modules = [
  Extension(
    'humanleague',
    sources=source_files(),
    include_dirs=[
      get_pybind_include(),
    ],
    depends=["setup.py", "DESCRIPTION", "src/docstr.inl"] + header_files(),
    language='c++'
  ),
]

class BuildExt(build_ext):
  """A custom build extension for adding compiler-specific options."""
  # c_opts = {
  #     'msvc': ['/EHsc'],
  #     'unix': [],
  # }
  # l_opts = {
  #     'msvc': [],
  #     'unix': [],
  # }

  # if sys.platform == 'darwin':
  #   darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.7']
  #   c_opts['unix'] += darwin_opts
  #   l_opts['unix'] += darwin_opts

  def build_extensions(self):
    ct = self.compiler.compiler_type

    # opts = self.c_opts.get(ct, [])
    # link_opts = self.l_opts.get(ct, [])
    # if ct == 'unix':
    #   if True: #has_flag(self.compiler, '-fvisibility=hidden'):
    #     opts.append('-fvisibility=hidden')

    for ext in self.extensions:
      ext.define_macros = defines(ct) 
      ext.extra_compile_args = cxxflags(ct)
      ext.extra_link_args = ldflags(ct)

    build_ext.build_extensions(self)

setup(
  name = 'humanleague',
  version = version(),
  description = 'Microsynthesis using quasirandom sampling and/or IPF',
  author = 'Andrew P Smith',
  author_email = 'a.p.smith@leeds.ac.uk',
  url = 'http://github.com/virgesmith/humanleague',
  long_description = readme(),
  long_description_content_type="text/markdown",
  ext_modules=ext_modules,
  cmdclass={'build_ext': BuildExt},
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
