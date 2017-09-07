
// Disabled for now
// TODO determine whether this is best practice
//#if 0


#include <R.h>
#include <Rinternals.h>
#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>

extern SEXP humanleague_prob2IntFreq(SEXP, SEXP);
extern SEXP humanleague_sobolSequence(SEXP, SEXP, SEXP);
extern SEXP humanleague_synthPop(SEXP);
extern SEXP humanleague_synthPopC(SEXP, SEXP);
extern SEXP humanleague_synthPopR(SEXP, SEXP);
extern SEXP humanleague_synthPopG(SEXP, SEXP);
extern SEXP humanleague_constrain(SEXP, SEXP);
extern SEXP humanleague_correlatedSobol2Sequence(SEXP, SEXP, SEXP);
extern SEXP humanleague_unitTest();

static const R_CallMethodDef CallEntries[] = {
    {"humanleague_prob2IntFreq",  (DL_FUNC) &humanleague_prob2IntFreq,  2},
    {"humanleague_sobolSequence", (DL_FUNC) &humanleague_sobolSequence, 3},
    {"humanleague_synthPop",      (DL_FUNC) &humanleague_synthPop,      1},
    {"humanleague_synthPopC",      (DL_FUNC) &humanleague_synthPopC,      2},
    {"humanleague_synthPopR",      (DL_FUNC) &humanleague_synthPopC,      2},
    {"humanleague_synthPopG",      (DL_FUNC) &humanleague_synthPopG,      2},
    {"humanleague_constrain",      (DL_FUNC) &humanleague_constrain,      2},
    {"humanleague_correlatedSobol2Sequence", (DL_FUNC) &humanleague_correlatedSobol2Sequence,      3},
    {"humanleague_unitTest",       (DL_FUNC) &humanleague_unitTest,      0},
    {NULL, NULL, 0}
};

void R_init_humanleague(DllInfo *dll)
{
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}

//#endif
