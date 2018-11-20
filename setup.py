#!/usr/bin/env python3

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

# seems that this will clean build every time, might make more sense to just have a lightweight wrapper & precompiled lib?
cppmodule = Extension(
  'humanleague',
  define_macros = [('MAJOR_VERSION', '2'),
                   ('MINOR_VERSION', '0'),
                   ('PATCH_VERSION', '4'),
                   ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')
                  ],
  extra_compile_args=['-Wall', '-std=c++11'],
  include_dirs = ['.', '/usr/include', '/usr/local/include'], # numpy include appended later
  sources = ['src/Sobol.cpp',
             'src/SobolImpl.cpp',
             'src/QIS.cpp',
             'src/QISI.cpp',
             'src/QIWS.cpp',
             'src/GQIWS.cpp',
             'src/StatFuncs.cpp',
             'src/NDArrayUtils.cpp',
             'src/Index.cpp',
             'src/Integerise.cpp',
             'src/UnitTester.cpp',
             'src/TestNDArray.cpp',
             'src/TestQIWS.cpp',
             'src/TestSobol.cpp',
             'src/TestStatFuncs.cpp',
             'src/TestIndex.cpp',
             'src/TestSlice.cpp',
             'src/TestReduce.cpp',
             'humanleague/Object.cpp',
             'humanleague/py_api.cpp'],
  # for now safer to put up with full rebuilds every time
  depends = ['humanleague/Object.h', 'humanleague/Array.h']
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

