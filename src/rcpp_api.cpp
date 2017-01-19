/**********************************************************************

Copyright 2017 The University of Leeds

This file is part of the R Microsim package.

Microsim is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Foobar is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

A copy of the GNU General Public License can be found in LICENCE.txt
It can also be found at <http://www.gnu.org/licenses/>.

**********************************************************************/

#include <Rcpp.h>
using namespace Rcpp;

#include "QRPF.h"
#include <vector>


//' Generate an n-D Sobol sequence containing @param samples samples, optionally skipping
//' @param dim the dimension of the sequence
//' @param skip (optional, default=false) skips 2^k samples (per dimension) where k is largest integer s.t. 2^k < samples
//' @export
// [[Rcpp::export]]
NumericMatrix sobolSeq(int dim, int samples, bool skip = false)
{
  static const double scale = 0.5 / (1ull<<63);

  NumericMatrix m(samples, dim);

  Sobol s(dim, skip ? samples : 0);

  for (int j = 0; j < samples; ++j)
    for (int i = 0; i < dim; ++i)
      m(j,i) = s() * scale;

  return m;
}

//' Generate a 2D population matrix
//'
//' @param marginal0 an integer vector containing marginal data
//' @param marginal1 an integer vector containing marginal data
//' @param maxAttempts (optional, default=4) number of retries to make if fitting is unsuccessful
//' @export
// [[Rcpp::export]]
IntegerMatrix pop(IntegerVector marginal0, IntegerVector marginal1, int maxAttempts = 4)
{
  std::vector<QRPF<2>::marginal_t> m;
  m.push_back(QRPF<2>::marginal_t(marginal0.begin(), marginal0.end())); // cols
  m.push_back(QRPF<2>::marginal_t(marginal1.begin(), marginal1.end())); // rows

  QRPF<2> qrpf(m);

  bool conv = qrpf.solve(maxAttempts);

  IntegerMatrix result(marginal0.size(), marginal1.size());

  if (!conv)
  {
    Rcout << "Failed to find exact solution after " << maxAttempts << " attempts.\n"
          << "Consider increasing maxAttempts (default=4)" << std::endl;
    return result; //unpopulated
  }

  Rcout << "Found solution with mean square variation of " << qrpf.msv() << std::endl;

  const QRPF<2>::table_t& t = qrpf.result();

  for (size_t j = 0; j < t.size(); ++j)
    for (size_t i = 0; i < t[0].size(); ++i)
      result(i,j) = t[j][i];

  return result;
}

//' Generate a 3D population as a list
//'
//' @param marginal0 an integer vector containing marginal data
//' @param marginal1 an integer vector containing marginal data
//' @param marginal2 an integer vector containing marginal data
//' @param maxAttempts (optional, default=4) number of retries to make if fitting is unsuccessful
//' @export
// [[Rcpp::export]]
DataFrame pop3(IntegerVector marginal0, IntegerVector marginal1,  IntegerVector marginal2, int maxAttempts = 4)
{
  std::vector<QRPF<2>::marginal_t> m;
  m.push_back(QRPF<2>::marginal_t(marginal0.begin(), marginal0.end())); // cols
  m.push_back(QRPF<2>::marginal_t(marginal1.begin(), marginal1.end())); // rows
  m.push_back(QRPF<2>::marginal_t(marginal2.begin(), marginal2.end())); // slices

  QRPF<3> qrpf(m);

  bool conv = qrpf.solve(maxAttempts);

  if (!conv)
  {
    Rcout << "Failed to find exact solution after " << maxAttempts << " attempts.\n"
          << "Consider increasing maxAttempts (default=4)" << std::endl;
    return IntegerMatrix(1,3);
  }

  Rcout << "Found solution with mean square variation of " << qrpf.msv() << std::endl;

  const QRPF<3>::table_t& t = qrpf.result();

  int id = 0;

  //DataFrame result = DataFrame::create(Named("id"), Named("c1"), Named("c2", Named("c3")));
  IntegerMatrix result(qrpf.sum(), 3);

  for (size_t j = 0; j < t.size(); ++j)
    for (size_t i = 0; i < t[0].size(); ++i)
      for (size_t k = 0; k < t[0][0].size(); ++k)
      {
        uint32_t p = t[j][i][k];
        while (p != 0)
        {
          //Rcout << p;
          //result(id, 0) = ++id;
          result(id, 0) = j;
          result(id, 1) = i;
          result(id, 2) = k;
          --p;
          ++id;
        }
      }
  return result;
}

