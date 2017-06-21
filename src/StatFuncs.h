
#pragma once

#include <array>
#include <utility>
#include <cstdint>


// Cumulative standard normal distribution function
double cumNorm(double x);

// Adapted from https://github.com/lballabio/QuantLib/blob/master/ql/math/distributions/normaldistribution.cpp
double invCumNorm(double x);

// Cholesky factorisation for single correlation (2x2)
std::array<double,4> cholesky(double rho);

// Chi-squared p-value calculation using incomplete gamma function
std::pair<double,bool> pValue(uint32_t df, double x);
