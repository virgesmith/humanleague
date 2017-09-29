
#pragma once


#include "NDArray.h"
#include "Index.h"

#include <vector>
#include <utility>
#include <cstdint>
#include <cmath>

// Cumulative standard normal distribution function
double cumNorm(double x);

// Adapted from https://github.com/lballabio/QuantLib/blob/master/ql/math/distributions/normaldistribution.cpp
double invCumNorm(double x);

// Cholesky factorisation for single correlation (2x2)
class Cholesky
{
public:
  explicit Cholesky(double rho);

  // generate correlated pair in [0,2^32) from uncorrelated pair in [0,2^32)
  // perhaps not the best type to pass in, but the most efficient
  std::pair<uint32_t, uint32_t> operator()(const std::vector<uint32_t>& uncorrelated) const;

private:
  // m_data[0] and m_data[1] are elements of Cholesky factorisation corresponding to rho and sqrt(1-rho*rho) respectively
  double m_data[2];
};

// chi-squared stat: sample vs reference (max entropy)
template<typename T, typename U>
double chiSq(const NDArray<T>& sample, const NDArray<U>& reference)
{
  double chisq = 0.0;
  for (Index index(sample.sizes()); !index.end(); ++index)
  {
    // m is the mean population of this state
    chisq += (sample[index] - reference[index]) * (sample[index] - reference[index]) / reference[index];
  }
  return chisq;
}

template<typename T>
inline double factorial(T x)
{
  return std::tgamma(x+1);
}

int64_t dof(std::vector<int64_t> sizes);

// Chi-squared p-value calculation using incomplete gamma function
std::pair<double,bool> pValue(uint32_t df, double x);

// S!/(prod_k(a_k!))
double degeneracy(const NDArray<int64_t>& a);
