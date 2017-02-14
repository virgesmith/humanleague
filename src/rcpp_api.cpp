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

#include "IWRS.h"
#include "QIPF.h"
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
  result["pValue"] = solver.pValue();
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

//' Generate a population in n dimensions given n marginals
//'
//' @param marginals a List of n integer vectors containing marginal data (2 <= n <= 12). The sum of elements in each vector must be identical
//' @export
// [[Rcpp::export]]
List synthPop(List marginals, const std::string& method = "iwrs")
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
  result["method"] = method;

  if (method == "iwrs")
  {
    // Workaround for fact that dimensionality is a template param and thus fixed at compile time
    switch(dim)
    {
    case 2:
      doSolve<IWRS<2>>(result, dims, m);
      break;
    case 3:
      doSolve<IWRS<3>>(result, dims, m);
      break;
    case 4:
      doSolve<IWRS<4>>(result, dims, m);
      break;
    case 5:
      doSolve<IWRS<5>>(result, dims, m);
      break;
    case 6:
      doSolve<IWRS<6>>(result, dims, m);
      break;
    case 7:
      doSolve<IWRS<7>>(result, dims, m);
      break;
    case 8:
      doSolve<IWRS<8>>(result, dims, m);
      break;
    case 9:
      doSolve<IWRS<9>>(result, dims, m);
      break;
    case 10:
      doSolve<IWRS<10>>(result, dims, m);
      break;
    case 11:
      doSolve<IWRS<11>>(result, dims, m);
      break;
    case 12:
      doSolve<IWRS<12>>(result, dims, m);
      break;
    default:
      throw std::runtime_error("invalid dimensionality: " + std::to_string(dim));
    }
  }
  else if (method == "qipf")
  {
    // Workaround for fact that dimensionality is a template param and thus fixed at compile time
    switch(dim)
    {
    case 2:
      doSolve<IWRS<2>>(result, dims, m);
      break;
    case 3:
      doSolve<IWRS<3>>(result, dims, m);
      break;
    case 4:
      doSolve<IWRS<4>>(result, dims, m);
      break;
    case 5:
      doSolve<IWRS<5>>(result, dims, m);
      break;
    case 6:
      doSolve<IWRS<6>>(result, dims, m);
      break;
    case 7:
      doSolve<IWRS<7>>(result, dims, m);
      break;
    case 8:
      doSolve<IWRS<8>>(result, dims, m);
      break;
    case 9:
      doSolve<IWRS<9>>(result, dims, m);
      break;
    case 10:
      doSolve<IWRS<10>>(result, dims, m);
      break;
    case 11:
      doSolve<IWRS<11>>(result, dims, m);
      break;
    case 12:
      doSolve<IWRS<12>>(result, dims, m);
      break;
    default:
      throw std::runtime_error("invalid dimensionality: " + std::to_string(dim));
    }
  }
  else
  {
    throw std::runtime_error("Invalid method. Valid values are: 'iwrs', 'qipf'");
  }
  // TODO dump out pop table...

  return result;
}

