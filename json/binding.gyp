{
  'targets': [
    {
      'target_name': 'humanleague',
      'cflags_cc': [ '-g -O2 -Wall -Werror -std=c++11' ],
      'cflags_cc!': [ '-fno-rtti', '-fno-exceptions' ],
      'sources': [ 'json_api.cpp',
                   'Module.cpp',
                   '../src/Sobol.cpp',
                   '../src/SobolImpl.c',
                   '../src/CQIWS.cpp',
                   '../src/RQIWS.cpp',
                   '../src/StatFuncs.cpp',
                   '../src/NDArrayUtils.cpp'],
      'include_dirs': ['../..'],  
    },
  ],
} 

