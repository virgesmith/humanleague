#!/usr/bin/env python3

import os
import numpy
from distutils.core import Extension, setup

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
             'humanleague/Object.cpp',
             'humanleague/py_api.cpp'],
  # annoyingly *.h causes a full rebuild every time
  depends = ['Object.h', 'Array.h', 'setup.py']
)

setup(
  name = 'humanleague',
  version = '0.0',
  description = 'microsynthesis using quasirandom sampling',
  author = 'Andrew Smith',
  author_email = 'a.p.smith@leeds.ac.uk',
  url = '',
  long_description = '''
microsynthesis using quasirandom sampling
''',
  ext_modules = [cppmodule],
)

