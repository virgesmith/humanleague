#pragma once

#include "NDArray.h"
#include "NDArrayUtils.h"

#include <algorithm>
#include <vector>
//#include <array> sadly cant use this (easily) as length is defined in type

#include <cmath>


template<size_t D>
class IPF 
{
public:

  static const size_t Dim = D;

  IPF(const NDArray<D, double>& seed, const std::vector<std::vector<double>>& marginals)
    : m_result(seed.sizes()), m_marginals(marginals), m_errors(D), m_conv(false)
  {
    m_errors[0].resize(m_marginals[0].size());
    m_errors[1].resize(m_marginals[1].size());
    print(m_marginals[0]);
    print(m_marginals[1]);
    print(reduce<2, double, 0>(seed));
    print(reduce<2, double, 1>(seed));
  
    std::copy(seed.rawData(), seed.rawData() + seed.storageSize(), m_result.begin());
    print(reduce<Dim, double, 0>(m_result));
    print(reduce<Dim, double, 1>(m_result));
    
    for (m_iters = 0; !m_conv && m_iters < s_MAXITER; ++m_iters) 
    {
      const std::vector<double>& r0 = reduce<Dim, double, 0>(m_result);
      for (size_t p = 0; p < m_marginals[0].size(); ++p)
      {
        Index<2, 0> index(m_result.sizes(), p);
        for (; !index.end(); ++index) 
        {
          m_result[index] *= m_marginals[0][index[0]] / r0[p]; 
        }
      }
      print(m_result.rawData(), m_result.storageSize(), m_marginals[1].size());
      
      const std::vector<double>& r1 = reduce<Dim, double, 1>(m_result);
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
      std::vector<std::vector<double>> diffs(Dim);
      // TODO template loop
      diffs[0] = reduce<Dim, double, 0>(m_result);
      diffs[1] = reduce<Dim, double, 1>(m_result);
      
      m_conv = computeErrors(diffs);
  
      print(m_result.rawData(), m_result.storageSize(), m_marginals[1].size());
    }
  }
  
  IPF(const IPF&) = delete;
  IPF(IPF&&) = delete;
  
  IPF& operator=(const IPF&) = delete;
  IPF& operator=(IPF&&) = delete;
  
  virtual ~IPF() { }

  const std::vector<std::vector<double>> errors() const 
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

  // TODO might be more efficient to directly calc errors (not sure we really need the sign of the values)
  template<size_t O>
  void calcResiduals(std::vector<std::vector<double>>& r)
  {
    calcResiduals<O-1>(r);
    r[O-1] = diff(reduce<Dim, double, O-1>(m_result), m_marginals[O-1]);
  }

  // this is close to repeating the above 
  bool computeErrors(std::vector<std::vector<double>>& diffs)
  {
    calcResiduals<Dim>(diffs);
    double maxError = -std::numeric_limits<double>::max();
    for (size_t d = 0; d < Dim; ++d)
    {
      for (size_t i = 0; i < diffs[d].size(); ++i)
      {
        double e = std::fabs(diffs[d][i]);
        m_errors[0][i] = e;
        maxError = std::max(maxError, e);
      }
    }
    return maxError < m_tol;
  }
  
  static const size_t s_MAXITER = 10;

  NDArray<Dim, double> m_result;
  std::vector<std::vector<double>> m_marginals;
  std::vector<std::vector<double>> m_errors;
  size_t m_iters;
  bool m_conv;
  const double m_tol = 1e-8;
  
};

// disable 
template<> class IPF<0>;
template<> class IPF<1>;
template<> class IPF<4>; //...

// TODO helper macro for member template specialisations
#define SPECIALISE_CALCRESIDUALS(d) \
template<> \
template<> \
inline void IPF<d>::calcResiduals<1>(std::vector<std::vector<double>>& r) \
{ \
  r[0] = diff(reduce<d, double, 0>(m_result), m_marginals[0]); \
}

SPECIALISE_CALCRESIDUALS(2)
SPECIALISE_CALCRESIDUALS(3)

#undef SPECIALISE_CALCRESIDUALS
