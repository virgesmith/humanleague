
#include <R.h>
#include <Rinternals.h>
#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>

extern SEXP _humanleague_flatten(SEXP, SEXP);
extern SEXP _humanleague_prob2IntFreq(SEXP, SEXP);
extern SEXP _humanleague_sobolSequence(SEXP, SEXP, SEXP);
extern SEXP _humanleague_ipf(SEXP, SEXP);
extern SEXP _humanleague_qis(SEXP, SEXP);
extern SEXP _humanleague_qisi(SEXP, SEXP);
//extern SEXP _humanleague_correlatedSobol2Sequence(SEXP, SEXP, SEXP);
extern SEXP _humanleague_unitTest();

// legacy
extern SEXP _humanleague_synthPop(SEXP);
extern SEXP _humanleague_synthPopG(SEXP, SEXP);

static const R_CallMethodDef CallEntries[] = {
  {"humanleague_flatten",       (DL_FUNC) &_humanleague_flatten,  2},
  {"humanleague_prob2IntFreq",  (DL_FUNC) &_humanleague_prob2IntFreq,  2},
  {"humanleague_sobolSequence", (DL_FUNC) &_humanleague_sobolSequence, 3},
  {"humanleague_ipf",           (DL_FUNC) &_humanleague_ipf,           2},
  {"humanleague_qis",           (DL_FUNC) &_humanleague_qis,           2},
  {"humanleague_qisi",          (DL_FUNC) &_humanleague_qisi,          2},
  {"humanleague_unitTest",      (DL_FUNC) &_humanleague_unitTest,      0},
  // legacy functions (v1.0 compat)
  {"humanleague_synthPop",      (DL_FUNC) &_humanleague_synthPop,      1},
  {"humanleague_synthPopG",     (DL_FUNC) &_humanleague_synthPopG,     2},
  {NULL, NULL, 0}
};

void R_init_humanleague(DllInfo *dll)
{
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, TRUE);
}

