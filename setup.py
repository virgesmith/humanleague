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

def list_files(dirs, exts, exclude=[]):
  print("list_files")
  files = []
  if isinstance(exclude, str):
    exclude = [exclude]
  for directory in dirs:
    for ext in exts:
      files.extend(glob.glob(os.path.join(directory, "*." + ext)))
  [f in files and files.remove(f) for f in exclude]
  return files

def cxxflags(platform):
  print("cxxflags")
  if platform == "unix":
    return [
      "-Wall",
      "-Werror",
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
    return ['/std:c++11', '/EHsc']
  else:
    return []

def ldflags(_platform):
  return []

def defines(platform):
  return [
    ("HUMANLEAGUE_VERSION", version()),
    ("PYTHON_MODULE", "")
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
    sources=list_files(['src'], ["cpp"], exclude=["src/rcpp_api.cpp", "src/RcppExports.cpp", "src/humanleague_init.c"]),
    include_dirs=[
      get_pybind_include(),
    ],
    depends=["DESCRIPTION"] + list_files(["src"], ["h"]),
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
      print(self.distribution.get_version())
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
