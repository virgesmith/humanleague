
#include <R.h>
#include <Rinternals.h>
#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>

extern SEXP _humanleague_flatten(SEXP, SEXP);
extern SEXP _humanleague_prob2IntFreq(SEXP, SEXP);
extern SEXP _humanleague_sobolSequence(SEXP, SEXP, SEXP);
extern SEXP _humanleague_ipf(SEXP, SEXP, SEXP);
extern SEXP _humanleague_qis(SEXP, SEXP, SEXP);
extern SEXP _humanleague_qisi(SEXP, SEXP, SEXP, SEXP);
extern SEXP _humanleague_integerise(SEXP);
extern SEXP _humanleague_unitTest();


static const R_CallMethodDef CallEntries[] = {
  {"humanleague_flatten",       (DL_FUNC) &_humanleague_flatten,  2},
  {"humanleague_prob2IntFreq",  (DL_FUNC) &_humanleague_prob2IntFreq,  2},
  {"humanleague_integerise",    (DL_FUNC) &_humanleague_integerise,  1},
  {"humanleague_sobolSequence", (DL_FUNC) &_humanleague_sobolSequence, 3},
  {"humanleague_ipf",           (DL_FUNC) &_humanleague_ipf,           3},
  {"humanleague_qis",           (DL_FUNC) &_humanleague_qis,           3},
  {"humanleague_qisi",          (DL_FUNC) &_humanleague_qisi,          4},
  {"humanleague_unitTest",      (DL_FUNC) &_humanleague_unitTest,      0},
  {NULL, NULL, 0}
};

void R_init_humanleague(DllInfo *dll)
{
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, TRUE);
}

