// Adapted from https://github.com/stevengj/nlopt/

#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdint.h>

#define MAXDIM 1111
#define MAXDEG 12

typedef struct SobolData_ 
{
  uint32_t sdim; /* dimension of sequence being generated */
  uint32_t *mdata; /* array of length 32 * sdim */
  uint32_t *m[32]; /* more convenient pointers to mdata, of direction #s */
  uint32_t *x; /* previous x = x_n, array of length sdim */
  uint32_t *b; /* position of fixed point in x[i] is after bit b[i] */
  uint32_t n; /* number of x's generated so far */
} SobolData;


SobolData* nlopt_sobol_create(uint32_t sdim);

void nlopt_sobol_destroy(SobolData* s);

int nlopt_sobol_next(SobolData* s, uint32_t* x);

void nlopt_sobol_skip(SobolData* s, uint32_t n, uint32_t* x);

#ifdef __cplusplus
}
#endif
