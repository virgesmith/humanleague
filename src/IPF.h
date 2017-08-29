#pragma once

#include "NDArray.h"
#include "NDArrayUtils.h"

#include <algorithm>
#include <vector>
#include <array>

#include <cmath>

template<size_t D>
class IPF 
{
public:

  static const size_t Dim = D;

  // TODO generalise to nD
  IPF(const NDArray<D, double>& seed, const std::array<std::vector<double>, D>& marginals)
    : m_result(seed.sizes()), m_marginals(marginals), m_conv(false)
  {
    m_errors[0].resize(m_marginals[0].size());
    m_errors[1].resize(m_marginals[1].size());
    print(m_marginals[0]);
    print(m_marginals[1]);
    //print(reduce<2, double, 0>(seed));
    //print(reduce<2, double, 1>(seed));
  
    std::copy(seed.rawData(), seed.rawData() + seed.storageSize(), m_result.begin());
    print(reduce<2, double, 0>(m_result));
    print(reduce<2, double, 1>(m_result));
  
    for (m_iters = 0; !m_conv && m_iters < s_MAXITER; ++m_iters) 
    {
      const std::vector<double>& r0 = reduce<2, double, 0>(m_result);
      // marginal index could be wrong
      for (size_t p = 0; p < m_marginals[0].size(); ++p)
      {
        Index<2, 0> index(m_result.sizes(), p);
        for (; !index.end(); ++index) 
        {
          m_result[index] *= m_marginals[0][index[0]] / r0[p]; 
        }
      }
      print(m_result.rawData(), m_result.storageSize(), m_marginals[1].size());
      
      const std::vector<double>& r1 = reduce<2, double, 1>(m_result);
      // 
      for (size_t p = 0; p < marginals[1].size(); ++p)
      {
        Index<2, 1> index(m_result.sizes(), p);
        for (; !index.end(); ++index) 
        {
          m_result[index] *= m_marginals[1][index[1]] / r1[p]; 
        }
      }
      
      // inefficient copying
      std::array<std::vector<double>, 2> d;
      d[0] = reduce<2, double, 0>(m_result);
      d[1] = reduce<2, double, 1>(m_result);
      
      m_conv = computeErrors(d);
  
      print(m_result.rawData(), m_result.storageSize(), m_marginals[1].size());
    }
  }
  
  IPF(const IPF&) = delete;
  IPF(IPF&&) = delete;
  
  IPF& operator=(const IPF&) = delete;
  IPF& operator=(IPF&&) = delete;
  
  virtual ~IPF() { }

  const std::array<std::vector<double>, D> errors() const 
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

  bool computeErrors(const std::array<std::vector<double>, D>& d)
  {
    double maxError = -std::numeric_limits<double>::max();
    for (size_t i = 0; i < d[0].size(); ++i)
    {
      double e = std::fabs(m_marginals[0][i] - d[0][i]);
      m_errors[0][i] = e;
      maxError = std::max(maxError, e);
    }
    for (size_t i = 0; i < d[1].size(); ++i)
    {
      double e = std::fabs(m_marginals[1][i] - d[1][i]);
      m_errors[1][i] = e;
      maxError = std::max(maxError, e);
    }
    return maxError < m_tol;
  }
  
  static const size_t s_MAXITER = 10;

  NDArray<Dim, double> m_result;
  std::array<std::vector<double>, Dim> m_marginals;
  std::array<std::vector<double>, Dim> m_errors;
  size_t m_iters;
  bool m_conv;
  const double m_tol = 1e-8;
  
};

// disable 
template<> class IPF<0>;
template<> class IPF<1>;
template<> class IPF<3>; //...