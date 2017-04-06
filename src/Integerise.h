
#pragma once

#include <vector>
#include <cstdlib>

// Given pop and real number densities (sum = 1), produce integer frequencies with minimal mean squuare error
std::vector<int> integeriseMarginalDistribution(const std::vector<double>& p, size_t pop, double& mse);
