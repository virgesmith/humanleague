#pragma once

// Quasirandomly sampled IPF

#include "IPF.h"
#include "Sobol.h"

// TODO move somewhere appropriate (doesnt need to be member)
inline size_t pick(const std::vector<double>& dist, double r)
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


template<size_t D>
void decrementMarginals(std::vector<std::vector<double>>& m, const size_t* index)
{
  --m[D-1][index[D-1]];
  decrementMarginals<D-1>(m, index);
// --(this->m_marginals[1][index[1]]);
// --(this->m_marginals[2][index[2]]);
}

// end the recursion
template<>
void decrementMarginals<1>(std::vector<std::vector<double>>& m, const size_t* index)
{
  --m[0][index[0]];
}


template<size_t D>
void getIndex(const old::NDArray<D, double>& p, const std::vector<uint32_t>& r, size_t* index)
{
  // TODO template bloat
  static const double scale = 0.5 / (1u<<31);

  // reduce dim D-1
  const std::vector<double>& m = old::reduce<D, double, D-1>(p);
  // pick an index
  index[D-1] = pick(m, r[D-1] * scale);

  // take slice of Dim D-1 at index
  old::NDArray<D-1, double> sliced = old::slice<D, double, D-1>(p, index[D-1]);

  // recurse
  getIndex<D-1>(sliced, r, index);
}

// end recursion
template<>
void getIndex<2>(const old::NDArray<2, double>& p, const std::vector<uint32_t>& r, size_t* index)
{
  // TODO template bloat
  static const double scale = 0.5 / (1u<<31);

  // reduce dim 1 (now 0)
  const std::vector<double>& r1 = old::reduce<2, double, 1>(p);
  // pick an index
  index[1] = pick(r1, r[1] * scale);

  // slice dim 2 (now 0)
  const std::vector<double>& r0 = old::slice<double, 1>(p, index[1]);
  // no reduction required
  // pick an index
  index[0] = pick(r0, r[0] * scale);
}


// TODO IPF should be member not super
template<size_t D>
class QSIPF : public IPF<D>
{
public:
  // TODO marginal values must be integers
  QSIPF(const old::NDArray<D, double>& seed, const std::vector<std::vector<int64_t>>& marginals)
  : IPF<D>(seed, marginals), m_sample(seed.sizes()), m_ipfSolution(seed.sizes()), m_recalcs(0)
  {
    m_originalPopulation = this->m_population;
    std::copy(this->m_result.rawData(), this->m_result.rawData() + this->m_result.storageSize(), const_cast<double*>(m_ipfSolution.rawData()));

    if (!this->m_conv)
      throw std::runtime_error("Initial IPF failed to converge, check seed and marginals");

    doit(seed);
  }

  ~QSIPF()
  {
  }

private:
  void doit(const old::NDArray<D, double>& seed)
  {
    // Sample without replacement of static IPF
    // the original population (IPF::m_population will reduce as we sample)
    m_sample.assign(0);
    Sobol qrng(D);
    //const double scale = 0.5 / (1u<<31);
    size_t index[D] = {0};
    for (size_t i = 0; i < m_originalPopulation; ++i)
    {
      if (!this->m_conv)
        throw std::runtime_error("IPF convergence failure");
      //const std::vector<uint32_t>& r = qrng.buf();

      getIndex<D>(this->m_result, qrng.buf(), &index[0]);

      for (size_t i = 0; i < D; ++i)
        if (index[i] >= this->m_result.sizes()[i])
          throw std::runtime_error("index out of bounds at " + std::to_string(i) + ": " + std::to_string(index[i]));

      // without replacement
      // directly decrementing the IPF population doesnt converge
      decrementMarginals<D>(this->m_marginals, index);

      ++m_sample[index];
      --this->m_result[index];
      // only recompute IPF solution when a probability goes negative
      if (this->m_result[index] < 0) 
      {
        ++m_recalcs;
        this->solve(seed);
        // give up if IPF doesnt converge
        if (!this->m_conv)
          break;
      }
    }
  }

public:
  const old::NDArray<D, int64_t>& sample() const
  {
    return m_sample;
  }

  virtual size_t population() const
  {
    return m_originalPopulation;
  }

  // This returns the number of times the IPF population was recalculated
  virtual size_t iters() const
  {
    return m_recalcs;
  }

  // chi-squared stat vs the IPF solution
  double chiSq() const 
  {
    double chisq = 0.0;
    for (old::Index<D, old::Index_Unfixed> index(m_sample.sizes()); !index.end(); ++index)
    {
      // m is the mean population of this state
      chisq += (m_sample[index] - m_ipfSolution[index]) * (m_sample[index] - m_ipfSolution[index]) / m_ipfSolution[index];
    }
    return chisq;
  }

private:

  size_t m_originalPopulation;
  old::NDArray<D, int64_t> m_sample;
  old::NDArray<D, double> m_ipfSolution;
  size_t m_recalcs;
  //double m_chi2;
};
