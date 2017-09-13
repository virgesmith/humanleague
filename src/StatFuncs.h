
#pragma once

#include <array>
#include <vector>
#include <utility>
#include <cstdint>


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

// Chi-squared p-value calculation using incomplete gamma function
std::pair<double,bool> pValue(uint32_t df, double x);
