
#pragma once

#include "QISI.h"

#include <vector>
#include <memory>
#include <cstdlib>

// Given pop and real number densities (sum = 1), produce integer frequencies with minimal mean squuare error
std::vector<int> integeriseMarginalDistribution(const std::vector<double>& p, int pop, double& mse);

// Given a fraction population in n dimensions, with integral marginal sums, construct a QISI object using the 1d marginals
std::unique_ptr<QISI> integerise_multidim(const NDArray<double>& seed);
