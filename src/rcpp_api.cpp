/**********************************************************************

Copyright 2017 The University of Leeds

This file is part of the R humanleague package.

humanleague is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

humanleague is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

A copy of the GNU General Public License can be found in LICENCE.txt
in the project's root directory, or at <http://www.gnu.org/licenses/>.

**********************************************************************/

#include <Rcpp.h>
using namespace Rcpp;

#include "QIWS.h"
//#include "IQRS.h"
#include "Integerise.h"
#include <vector>

//#include <csignal>

// // Handler for ctrl-C
// extern "C" void sigint_handler(int)
// {
//   // throw back to R
//   throw std::runtime_error("User interrupt");

// }
// Enable ctrl-C to interrupt the code
// TODO this doesnt seem to work, perhaps another approach (like a separate thread?)
//void (*oldhandler)(int) = signal(SIGINT, sigint_handler);


template<typename S>
void doSolve(List& result, IntegerVector dims, const std::vector<std::vector<uint32_t>>& m)
{
  S solver(m);
  result["conv"] = solver.solve();
  result["chiSq"] = solver.chiSq();
  std::pair<double, bool> pVal = solver.pValue();
  result["pValue"] = pVal.first;
  if (!pVal.second)
  {
    result["warning"] = "p-value may be inaccurate";
  }
  result["error.margins"] = std::vector<uint32_t>(solver.residuals(), solver.residuals() + S::Dim);
  const typename S::table_t& t = solver.result();
  const NDArray<S::Dim, double>& p = solver.stateProbabilities();
  Index<S::Dim, Index_Unfixed> idx(t.sizes());
  IntegerVector values(t.storageSize());
  NumericVector probs(t.storageSize());
  while (!idx.end())
  {
    values[idx.colMajorOffset()] = t[idx];
    probs[idx.colMajorOffset()] = p[idx];
    ++idx;
  }
  values.attr("dim") = dims;
  probs.attr("dim") = dims;
  result["p.hat"] = probs;
  result["x.hat"] = values;
}

//' Generate a population in n dimensions given n marginals.
//'
//' Using Quasirandom Integer Without-replacement Sampling (QIWS), this function
//' generates an n-dimensional population table where elements sum to the input marginals, and supplemental data.
//' @param marginals a List of n integer vectors containing marginal data (2 <= n <= 12). The sum of elements in each vector must be identical
//' @return an object containing: the population matrix, the occupancy probability matrix, a convergence flag, the chi-squared statistic, p-value, and error value (nonzero if not converged)
//' @examples
//' synthPop(list(c(1,2,3,4), c(3,4,3)))
//' @export
// [[Rcpp::export]]
List synthPop(List marginals)
{
  const size_t dim = marginals.size();
  std::vector<std::vector<uint32_t>> m(dim);
  IntegerVector dims;
  for (size_t i = 0; i < dim; ++i)
  {
    const IntegerVector& iv = marginals[i];
    m[i].reserve(iv.size());
    std::copy(iv.begin(), iv.end(), std::back_inserter(m[i]));
    dims.push_back(iv.size());
  }
  List result;
  result["method"] = "QIWS";

  // Workaround for fact that dimensionality is a template param and thus fixed at compile time
  switch(dim)
  {
  case 2:
    doSolve<QIWS<2>>(result, dims, m);
    break;
  case 3:
    doSolve<QIWS<3>>(result, dims, m);
    break;
  case 4:
    doSolve<QIWS<4>>(result, dims, m);
    break;
  case 5:
    doSolve<QIWS<5>>(result, dims, m);
    break;
  case 6:
    doSolve<QIWS<6>>(result, dims, m);
    break;
  case 7:
    doSolve<QIWS<7>>(result, dims, m);
    break;
  case 8:
    doSolve<QIWS<8>>(result, dims, m);
    break;
  case 9:
    doSolve<QIWS<9>>(result, dims, m);
    break;
  case 10:
    doSolve<QIWS<10>>(result, dims, m);
    break;
  case 11:
    doSolve<QIWS<11>>(result, dims, m);
    break;
  case 12:
    doSolve<QIWS<12>>(result, dims, m);
    break;
  default:
    throw std::runtime_error("invalid dimensionality: " + std::to_string(dim));
  }

  return result;
}


//' Generate integer frequencies from discrete probabilities and an overall population.
//'
//' This function will generate the closest integer vector to the probabilities scaled to the population.
//' @param pIn a numeric vector of state occupation probabilities. Must sum to unity (to within double precision epsilon)
//' @param pop the total population
//' @return an integer vector of frequencies that sums to pop.
//' @examples
//' prob2IntFreq(c(0.1,0.2,0.3,0.4), 11)
//' @export
// [[Rcpp::export]]
List prob2IntFreq(NumericVector pIn, int pop)
{
  double var;
  const std::vector<double>& p = as<std::vector<double>>(pIn);

  if (pop < 1)
  {
    throw std::runtime_error("population must be strictly positive");
  }

  if (fabs(std::accumulate(p.begin(), p.end(), -1.0)) > std::numeric_limits<double>::epsilon())
  {
    throw std::runtime_error("probabilities do not sum to unity");
  }
  std::vector<int> f = integeriseMarginalDistribution(p, pop, var);

  List result;
  result["freq"] = f;
  result["var"] = var;

  return result;
}

//' Generate Sobol' quasirandom sequence
//'
//' @param dim dimensions
//' @param n number of variates to sample
//' @param skip number of variates to skip (actual number skipped will be largest power of 2 less than k)
//' @return a n-by-d matrix of uniform probabilities in (0,1).
//' @examples
//' sobolSequence(2, 1000, 1000) # will skip 512 numbers!
//' @export
// [[Rcpp::export]]
NumericMatrix sobolSequence(int dim, int n, int skip = 0)
{
  static const double scale = 0.5 / (1ull<<31);

  NumericMatrix m(n, dim);

  Sobol s(dim, skip);

  for (int j = 0; j <n ; ++j)
    for (int i = 0; i < dim; ++i)
      m(j,i) = s() * scale;

  return m;
}
