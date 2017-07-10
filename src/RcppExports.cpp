// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

// synthPop
List synthPop(List marginals);
RcppExport SEXP humanleague_synthPop(SEXP marginalsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type marginals(marginalsSEXP);
    rcpp_result_gen = Rcpp::wrap(synthPop(marginals));
    return rcpp_result_gen;
END_RCPP
}
// synthPopC
List synthPopC(List marginals, LogicalMatrix permittedStates);
RcppExport SEXP humanleague_synthPopC(SEXP marginalsSEXP, SEXP permittedStatesSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type marginals(marginalsSEXP);
    Rcpp::traits::input_parameter< LogicalMatrix >::type permittedStates(permittedStatesSEXP);
    rcpp_result_gen = Rcpp::wrap(synthPopC(marginals, permittedStates));
    return rcpp_result_gen;
END_RCPP
}
// synthPopG
List synthPopG(List marginals, NumericMatrix exoProbsIn);
RcppExport SEXP humanleague_synthPopG(SEXP marginalsSEXP, SEXP exoProbsInSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type marginals(marginalsSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type exoProbsIn(exoProbsInSEXP);
    rcpp_result_gen = Rcpp::wrap(synthPopG(marginals, exoProbsIn));
    return rcpp_result_gen;
END_RCPP
}
// synthPopR
List synthPopR(List marginals, double rho);
RcppExport SEXP humanleague_synthPopR(SEXP marginalsSEXP, SEXP rhoSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type marginals(marginalsSEXP);
    Rcpp::traits::input_parameter< double >::type rho(rhoSEXP);
    rcpp_result_gen = Rcpp::wrap(synthPopR(marginals, rho));
    return rcpp_result_gen;
END_RCPP
}
// constrain
List constrain(IntegerMatrix population, LogicalMatrix permittedStates);
RcppExport SEXP humanleague_constrain(SEXP populationSEXP, SEXP permittedStatesSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< IntegerMatrix >::type population(populationSEXP);
    Rcpp::traits::input_parameter< LogicalMatrix >::type permittedStates(permittedStatesSEXP);
    rcpp_result_gen = Rcpp::wrap(constrain(population, permittedStates));
    return rcpp_result_gen;
END_RCPP
}
// prob2IntFreq
List prob2IntFreq(NumericVector pIn, int pop);
RcppExport SEXP humanleague_prob2IntFreq(SEXP pInSEXP, SEXP popSEXP) {
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
RcppExport SEXP humanleague_sobolSequence(SEXP dimSEXP, SEXP nSEXP, SEXP skipSEXP) {
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
RcppExport SEXP humanleague_correlatedSobol2Sequence(SEXP rhoSEXP, SEXP nSEXP, SEXP skipSEXP) {
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
RcppExport SEXP humanleague_unitTest() {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(unitTest());
    return rcpp_result_gen;
END_RCPP
}
