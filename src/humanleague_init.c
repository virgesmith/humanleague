#include <R.h>
#include <Rinternals.h>
#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>

extern SEXP humanleague_prob2IntFreq(SEXP, SEXP);
extern SEXP humanleague_sobolSequence(SEXP, SEXP, SEXP);
extern SEXP humanleague_synthPop(SEXP);

static const R_CallMethodDef CallEntries[] = {
    {"humanleague_prob2IntFreq",  (DL_FUNC) &humanleague_prob2IntFreq,  2},
    {"humanleague_sobolSequence", (DL_FUNC) &humanleague_sobolSequence, 3},
    {"humanleague_synthPop",      (DL_FUNC) &humanleague_synthPop,      1},
    {NULL, NULL, 0}
};

void R_init_humanleague(DllInfo *dll)
{
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
