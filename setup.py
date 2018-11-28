#!/usr/bin/env python3

import os
import glob
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

def readme():
  with open('README.md') as f:
    return f.read()

# Workaround for setup dependency on numpy
# delays getting the numpy include dir until numpy has been installed
class BuildExtNumpyWorkaround(build_ext):
  def run(self):
    import numpy
    # Add numpy headers to include_dirs
    self.include_dirs.append(numpy.get_include())
    # Call original build_ext command
    build_ext.run(self)

# this is far from ideal - checking the OS to imply the compiler
def platform_specific_compile_args():
  if os.name == 'nt':
    return ['/std:c++11']
  else:
    return ['-Wall', '-std=c++11']

def list_files(dirs, exts, exclude=[]):
  files = []
  for directory in dirs:
    for ext in exts:
      files.extend(glob.glob(os.path.join(directory, "*." + ext)))
  [files.remove(f) for f in exclude]
  return files

# seems that this will clean build every time, might make more sense to just have a lightweight wrapper & precompiled lib?
cppmodule = Extension(
  'humanleague',
  define_macros = [('MAJOR_VERSION', '2'),
                   ('MINOR_VERSION', '0'),
                   ('PATCH_VERSION', '4'),
                   ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')
                  ],
  extra_compile_args=platform_specific_compile_args(),
  include_dirs = ['.', '/usr/include', '/usr/local/include'], # numpy include appended later
  sources = list_files(["src", "humanleague"], ["cpp"], exclude=[os.path.join("src", "rcpp_api.cpp"), os.path.join("src", "RcppExports.cpp")]),
  # for now safer to put up with full rebuilds every time
  depends = list_files(["src", "humanleague"], ["h"])
)

import unittest
def test_suite():
  test_loader = unittest.TestLoader()
  test_suite = test_loader.discover('tests', pattern='test_*.py')
  return test_suite

#setuptools.
setup(
  name = 'humanleague',
  version = '2.0.4',
  description = 'Microsynthesis using quasirandom sampling and/or IPF',
  author = 'Andrew P Smith',
  author_email = 'a.p.smith@leeds.ac.uk',
  url = 'http://github.com/virgesmith/humanleague',
  long_description = readme(),
  long_description_content_type="text/markdown",
  cmdclass = {'build_ext': BuildExtNumpyWorkaround},
  ext_modules = [cppmodule],
  setup_requires=['numpy'],
  install_requires=['numpy'],
  test_suite='setup.test_suite'
)

