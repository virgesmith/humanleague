
// Disabled for now
// TODO determine whether this is best practice
//#if 0


#include <R.h>
#include <Rinternals.h>
#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>

extern SEXP _humanleague_prob2IntFreq(SEXP, SEXP);
extern SEXP _humanleague_sobolSequence(SEXP, SEXP, SEXP);
extern SEXP _humanleague_synthPop(SEXP);
//extern SEXP _humanleague_synthPopC(SEXP, SEXP);
//extern SEXP _humanleague_synthPopR(SEXP, SEXP);
//extern SEXP _humanleague_synthPopG(SEXP, SEXP);
extern SEXP _humanleague_ipf(SEXP, SEXP);
extern SEXP _humanleague_qsipf(SEXP, SEXP);
//extern SEXP _humanleague_constrain(SEXP, SEXP);
extern SEXP _humanleague_correlatedSobol2Sequence(SEXP, SEXP, SEXP);
extern SEXP _humanleague_unitTest();
//extern SEXP _humanleague_test(SEXP);

static const R_CallMethodDef CallEntries[] = {
  {"humanleague_prob2IntFreq",  (DL_FUNC) &_humanleague_prob2IntFreq,  2},
  {"humanleague_sobolSequence", (DL_FUNC) &_humanleague_sobolSequence, 3},
  {"humanleague_synthPop",      (DL_FUNC) &_humanleague_synthPop,      1},
//  {"humanleague_synthPopC",      (DL_FUNC) &_humanleague_synthPopC,      2},
//  {"humanleague_synthPopR",      (DL_FUNC) &_humanleague_synthPopC,      2},
//  {"humanleague_synthPopG",      (DL_FUNC) &_humanleague_synthPopG,      2},
  {"humanleague_ipf",            (DL_FUNC) &_humanleague_ipf,            2},
  {"humanleague_qsipf",            (DL_FUNC) &_humanleague_qsipf,            2},
//  {"humanleague_constrain",      (DL_FUNC) &_humanleague_constrain,      2},
  {"humanleague_correlatedSobol2Sequence", (DL_FUNC) &_humanleague_correlatedSobol2Sequence,      3},
  {"humanleague_unitTest",       (DL_FUNC) &_humanleague_unitTest,      0},
  //{"humanleague_test",       (DL_FUNC) &_humanleague_test,      1},
  {NULL, NULL, 0}
};

void R_init_humanleague(DllInfo *dll)
{
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, TRUE);
}

//#endif
