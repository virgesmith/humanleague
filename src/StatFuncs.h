
#pragma once

#include <array>
#include <vector>
#include <utility>
#include <cstdint>


// Cumulative standard normal distribution function
double cumNorm(double x);

// Adapted from https://github.com/lballabio/QuantLib/blob/master/ql/math/distributions/normaldistribution.cpp
double invCumNorm(double x);

class Cholesky
{
public:
  Cholesky(double rho);

private:
  double m_data[2];
};

// Cholesky factorisation for single correlation (2x2)
std::array<double,4> cholesky(double rho);

// generate correlated pair in [0,2^32) from uncorrelated pair in [0,2^32)
// c01 and c11 are elements of Cholesky factorisation corresponding to rho and sqrt(1-rho*rho) respectively
std::vector<uint32_t> correlatePair(const std::vector<uint32_t>& u, double c01, double c11);

// Chi-squared p-value calculation using incomplete gamma function
std::pair<double,bool> pValue(uint32_t df, double x);
