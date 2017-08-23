#!/usr/bin/env python3

import os
import numpy
from distutils.core import Extension, setup
import distutils_pytest

# seems that this will clean build every time, might make more sense to just have a lightweight wrapper & precompiled lib?
cppmodule = Extension(
  'humanleague',
  define_macros = [('MAJOR_VERSION', '1'),
                   ('MINOR_VERSION', '0'),
                   ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')
                  ],
  extra_compile_args=['-Wall', '-std=c++11'],
  include_dirs = ['.', '/usr/include', '/usr/local/include', numpy.get_include()],
#             libraries = [':humanleague.so'],
#             library_dirs = ['/usr/local/lib','../src'],
  sources = ['src/Sobol.cpp',
             'src/SobolImpl.c',
             'src/RQIWS.cpp',
             'src/GQIWS.cpp',
             'src/StatFuncs.cpp',
             'src/NDArrayUtils.cpp',
             'src/Integerise.cpp',
             'humanleague/Object.cpp',
             'humanleague/py_api.cpp'],
  # TODO work out how to include all header deps without always triggering a full rebuild
  # for now safer to put up with full rebuilds every time
  depends = ['Object.h', 'Array.h']
)

setup(
  name = 'humanleague',
  version = '1.0.1',
  description = 'microsynthesis using quasirandom sampling',
  author = 'Andrew Smith',
  author_email = 'a.p.smith@leeds.ac.uk',
  url = '',
  long_description = '''
microsynthesis using quasirandom sampling
''',
  ext_modules = [cppmodule],
  # these settings appear not to be required
#  tests_require=['nose'],
#  test_suite='tests',
)

