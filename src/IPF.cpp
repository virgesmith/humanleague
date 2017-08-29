
#include "IPF.h"

#include "NDArrayUtils.h"

#include <algorithm>
#include <cmath>

IPF::IPF(const NDArray<2, double>& seed, const std::array<std::vector<double>, 2>& marginals) 
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
        //m_result[index0] *= m_marginals[0][index0[0]] / r0[index0[0]]; 
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
        //m_result[index0] *= m_marginals[0][index0[0]] / r0[index0[0]]; 
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

bool IPF::computeErrors(const std::array<std::vector<double>, 2>& d)
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

  
// private:
//   NDArray<2, double> m_result;
//   std::array<std::vector<double>, 2> m_errors;
  
// };