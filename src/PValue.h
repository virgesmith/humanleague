
#pragma once

#include <utility>
#include <cstdint>

// Chi-squared p-value calculation using incomplete gamma function
std::pair<double,bool> pValue(uint32_t df, double x);
