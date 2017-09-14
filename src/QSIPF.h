#pragma once

// Quasirandomly sampled IPF

#include "IPF.h"
#include "Sobol.h"

template<size_t D>
class QSIPF : public IPF<D>
{
public:
  QSIPF(const NDArray<D, double>& seed, const std::vector<std::vector<double>>& marginals)
  : IPF<D>(seed, marginals), m_sample(seed.sizes()) 
  //m_result(seed.sizes()), m_marginals(marginals), m_errors(D), m_conv(false)
  {
    if (!this->m_conv)
      throw std::runtime_error("Initial IPF failed to converge, check seed and marginals");
    doit(seed);
  }

  ~QSIPF()
  {

  }

  void doit(const NDArray<D, double>& seed)
  {
    // Sample without replacement of static IPF 
    // n is the original population (this->population will reduce as we sample)
    const size_t n = this->m_population;
    m_sample.assign(0);
    Sobol qrng(D);
    const double scale = 0.5 / (1u<<31); 
    for (size_t i = 0; i < n; ++i)
    {
      this->solve(seed);
      if (!this->m_conv)
        throw std::runtime_error("IPF convergence failure");
      const std::vector<uint32_t>& r = qrng.buf();
      size_t index[D] = {0};
  
      // reduce dim 0
      const std::vector<double>& r0 = reduce<D, double, 0>(this->m_result);
      // pick an index
      index[0] = pick(r0, r[0] * scale);
  
      // take slice of Dim 0 at index 
      NDArray<D-1, double> slice0 = slice<D, double, 0>(this->m_result, index[0]);
      // reduce dim 1 (now 0)
      const std::vector<double>& r1 = reduce<D-1, double, 0>(slice0);
      // pick an index
      index[1] = pick(r1, r[1] * scale);
  
      // slice dim 2 (now 0)
      const std::vector<double>& r2 = slice<double, 0>(slice0, index[1]);
      // no reduction required
      // pick an index
      index[2] = pick(r2, r[2] * scale);
  
      // without replacement
      --(this->m_marginals[0][index[0]]);
      --(this->m_marginals[1][index[1]]);
      --(this->m_marginals[2][index[2]]);
      
      ++m_sample[index];
      //print(index, 3);
    }
  }

  const NDArray<D, uint32_t>& sample() const 
  {
    return m_sample;
  }

private:
  // TODO move somewhere appropriate (doesnt need to be member)
  size_t pick(const std::vector<double>& dist, double r)
  {
    // sum of dist should be 1, but we relax this
    // r is in (0,1) so scale up r by sum of dist
    r *= std::accumulate(dist.begin(), dist.end(), 0.0);
    double runningSum = 0.0;
    for (size_t i = 0; i < dist.size(); ++i)
    {
      runningSum += dist[i];
      if (r < runningSum)
        return i;
    }
    throw std::runtime_error("pick failed");

  }

  NDArray<D, uint32_t> m_sample;

};