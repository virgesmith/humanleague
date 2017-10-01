// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

// synthPop
List synthPop(List marginals);
RcppExport SEXP _humanleague_synthPop(SEXP marginalsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type marginals(marginalsSEXP);
    rcpp_result_gen = Rcpp::wrap(synthPop(marginals));
    return rcpp_result_gen;
END_RCPP
}
// synthPopG
List synthPopG(List marginals, NumericMatrix exoProbsIn);
RcppExport SEXP _humanleague_synthPopG(SEXP marginalsSEXP, SEXP exoProbsInSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type marginals(marginalsSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type exoProbsIn(exoProbsInSEXP);
    rcpp_result_gen = Rcpp::wrap(synthPopG(marginals, exoProbsIn));
    return rcpp_result_gen;
END_RCPP
}
// ipf
List ipf(NumericVector seed, List indices, List marginals);
RcppExport SEXP _humanleague_ipf(SEXP seedSEXP, SEXP indicesSEXP, SEXP marginalsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< List >::type indices(indicesSEXP);
    Rcpp::traits::input_parameter< List >::type marginals(marginalsSEXP);
    rcpp_result_gen = Rcpp::wrap(ipf(seed, indices, marginals));
    return rcpp_result_gen;
END_RCPP
}
// qis
List qis(List indices, List marginals);
RcppExport SEXP _humanleague_qis(SEXP indicesSEXP, SEXP marginalsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type indices(indicesSEXP);
    Rcpp::traits::input_parameter< List >::type marginals(marginalsSEXP);
    rcpp_result_gen = Rcpp::wrap(qis(indices, marginals));
    return rcpp_result_gen;
END_RCPP
}
// qsipf
List qsipf(NumericVector seed, List marginals);
RcppExport SEXP _humanleague_qsipf(SEXP seedSEXP, SEXP marginalsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< List >::type marginals(marginalsSEXP);
    rcpp_result_gen = Rcpp::wrap(qsipf(seed, marginals));
    return rcpp_result_gen;
END_RCPP
}
// prob2IntFreq
List prob2IntFreq(NumericVector pIn, int pop);
RcppExport SEXP _humanleague_prob2IntFreq(SEXP pInSEXP, SEXP popSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type pIn(pInSEXP);
    Rcpp::traits::input_parameter< int >::type pop(popSEXP);
    rcpp_result_gen = Rcpp::wrap(prob2IntFreq(pIn, pop));
    return rcpp_result_gen;
END_RCPP
}
// sobolSequence
NumericMatrix sobolSequence(int dim, int n, int skip);
RcppExport SEXP _humanleague_sobolSequence(SEXP dimSEXP, SEXP nSEXP, SEXP skipSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type dim(dimSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< int >::type skip(skipSEXP);
    rcpp_result_gen = Rcpp::wrap(sobolSequence(dim, n, skip));
    return rcpp_result_gen;
END_RCPP
}
// correlatedSobol2Sequence
NumericMatrix correlatedSobol2Sequence(double rho, int n, int skip);
RcppExport SEXP _humanleague_correlatedSobol2Sequence(SEXP rhoSEXP, SEXP nSEXP, SEXP skipSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type rho(rhoSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< int >::type skip(skipSEXP);
    rcpp_result_gen = Rcpp::wrap(correlatedSobol2Sequence(rho, n, skip));
    return rcpp_result_gen;
END_RCPP
}
// unitTest
List unitTest();
RcppExport SEXP _humanleague_unitTest() {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(unitTest());
    return rcpp_result_gen;
END_RCPP
}
