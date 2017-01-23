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

#include "QIPF.h"
#include <vector>


template<size_t D>
void doQipf(List& result, IntegerVector dims, const std::vector<std::vector<uint32_t>>& m, size_t maxAttempts)
{
  QIPF<D> qipf(m);
  result["conv"] = qipf.solve(maxAttempts);
  result["attempts"] = qipf.attempts();
  result["meanSqVariation"] = qipf.msv();
  const typename QIPF<D>::table_t& t = qipf.result();
  Index<D, Index_Unfixed> idx(t.sizes());
  IntegerVector values(t.storageSize());
  while (!idx.end())
  {
    values[idx.colMajorOffset()] = t[idx];
    ++idx;
  }
  Rcout << std::endl;
  values.attr("dim") = dims;
  result["x.hat"] = values;
}


//' Generate a population in n dimensions given n marginals
//'
//' @param marginals a List of n integer vectors containing marginal data
//' @param maxAttempts (optional, default=4) number of retries to make if fitting is unsuccessful
//' @export
// [[Rcpp::export]]
List synthPop(List marginals, int maxAttempts = 4)
{
  std::vector<std::vector<uint32_t>> m;
  IntegerVector dims;
  for (size_t i = 0; i < marginals.size(); ++i)
  {
    const IntegerVector& iv = marginals[i];
    std::vector<uint32_t> tmp;
    tmp.reserve(iv.size());
    std::copy(iv.begin(), iv.end(), std::back_inserter(tmp));
    m.push_back(tmp); // cols
    dims.push_back(iv.size());
  }
  const size_t dim = marginals.size();


  List result;
  result["method"] = "qipf";

  // Workaround for fact that QIPF dimensionality is a template param and thus fixed at compile time
  switch(dim)
  {
  case 2:
    doQipf<2>(result, dims, m, maxAttempts);
    break;
  case 3:
    doQipf<3>(result, dims, m, maxAttempts);
    break;
  case 4:
    doQipf<4>(result, dims, m, maxAttempts);
    break;
  case 5:
    doQipf<5>(result, dims, m, maxAttempts);
    break;
  case 6:
    doQipf<6>(result, dims, m, maxAttempts);
    break;
  case 7:
    doQipf<7>(result, dims, m, maxAttempts);
    break;
  case 8:
    doQipf<8>(result, dims, m, maxAttempts);
    break;
  case 9:
    doQipf<9>(result, dims, m, maxAttempts);
    break;
  case 10:
    doQipf<10>(result, dims, m, maxAttempts);
    break;
  case 11:
    doQipf<11>(result, dims, m, maxAttempts);
    break;
  case 12:
    doQipf<12>(result, dims, m, maxAttempts);
    break;
  default:
    throw std::runtime_error("invalid dimensionality: " + std::to_string(dim));
  }

  return result;
}

