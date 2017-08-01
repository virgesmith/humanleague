#!/usr/bin/python3

import os
import numpy
from distutils.core import setup, Extension

devroot = os.environ['HOME'] + "/dev"
#print("Project devroot: " + devroot)

module1 = Extension('humanleague',
                    define_macros = [('MAJOR_VERSION', '1'),
                                     ('MINOR_VERSION', '0'),
                                     ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')
                                     ],
                   extra_compile_args=['-Wall', '-std=c++11'],
# TODO env vars rather than relative paths....
                   include_dirs = [devroot, '/usr/include' '/usr/local/include', numpy.get_include()],
#                    libraries = [':humanleague.so'],
#                    library_dirs = ['/usr/local/lib','../src'],
                   sources = ['../src/Sobol.cpp',
                              '../src/SobolImpl.c',
                              '../src/RQIWS.cpp',
                              '../src/GQIWS.cpp',
                              '../src/StatFuncs.cpp',
                              '../src/NDArrayUtils.cpp',
                              'Object.cpp',
                              'py_api.cpp'],
# annoyingly *.h causes a full rebuild every time
                   depends = ['Object.h', 'Array.h', 'setup.py'])
# devroot + '/CatchDat/humanleague/src/*.h', 

# TODO unit test module?

setup (name = 'humanleague',
       version = '0.0',
       description = 'This is a demo package',
       author = '',
       author_email = '',
       url = '',
       long_description = '''
This is really just a demo package.
''',
       ext_modules = [module1])

