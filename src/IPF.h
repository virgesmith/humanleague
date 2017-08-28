#pragma once

#include "NDArray.h"

#include <vector>
#include <array>

class IPF 
{
public:
  IPF(const NDArray<2, double>& seed, const std::array<std::vector<double>, 2>& marginals);

  IPF(const IPF&) = delete;
  IPF(IPF&&) = delete;
  
  IPF& operator=(const IPF&) = delete;
  IPF& operator=(IPF&&) = delete;
  
  virtual ~IPF() { }

  const std::array<std::vector<double>, 2> errors() const 
  {
    return m_errors;
  }

  bool conv() const 
  {
    return m_conv;
  }

  size_t iters() const 
  {
    return m_iters;
  }
  
private:

  bool computeErrors(const std::array<std::vector<double>, 2>& d);

  static const size_t s_MAXITER = 10;

  NDArray<2, double> m_result;
  std::array<std::vector<double>, 2> m_marginals;
  std::array<std::vector<double>, 2> m_errors;
  size_t m_iters;
  bool m_conv;
  const double m_tol = 1e-8;
  
};