#pragma once

#include "NDArray.h"

#include <vector>
#include <array>

class IPF 
{
public:
  IPF(const NDArray<2, double>& seed, const std::array<std::vector<double>, 2>& marginals);

  IPF(const IPF&) = delete;

  IPF& operator=(const IPF&) = delete;

  virtual ~IPF() { }
  
private:
  NDArray<2, double> m_result;
  std::array<std::vector<double>, 2> m_errors;
  
};