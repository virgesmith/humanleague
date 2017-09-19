// Quasirandomly sampled IPF
#include "QSIPF.h"
#include "Index.h"
#include "NDArrayUtils.h"
#include "Sobol.h"

namespace {

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

void decrementMarginals(std::vector<std::vector<double>>& m, const Index& index)
{
  for (size_t d = 0; d < m.size(); ++d)
    --m[d][index[d]];
}

void getIndex(const NDArray<double>& p, const std::vector<uint32_t>& r, Index& index)
{
  static const double scale = 0.5 / (1u<<31);

  size_t dim = p.dim();

  if (dim > 2)
  {
    // reduce dim D-1
    const std::vector<double>& m = reduce<double>(p, dim - 1);
    // pick an index
    index[dim-1] = pick(m, r[dim-1] * scale);

    // take slice of Dim D-1 at index
    const NDArray<double>& sliced = slice<double>(p, {dim-1, index[dim-1]});
    
    // recurse
    getIndex(sliced, r, index);
  } 
  else
  {
    // reduce dim 1 (now 0)
    const std::vector<double>& r1 = reduce<double>(p, 1);
    // pick an index
    index[1] = pick(r1, r[1] * scale);

    // slice dim 2 (now 0)
    const NDArray<double>& sliced = slice<double>(p, {1, index[1]});
    assert(sliced.dim() == 1);
    std::vector<double> r0(sliced.rawData(), sliced.rawData() + sliced.storageSize());
    // no reduction required
    // pick an index
    index[0] = pick(r0, r[0] * scale);
  }
}

}

QSIPF::QSIPF(const NDArray<double>& seed, const std::vector<std::vector<int64_t>>& marginals)
: IPF(seed, marginals), m_sample(seed.sizes()), m_ipfSolution(seed.sizes()), m_recalcs(0)
{
  m_originalPopulation = this->m_population;
  std::copy(this->m_result.rawData(), this->m_result.rawData() + this->m_result.storageSize(), const_cast<double*>(m_ipfSolution.rawData()));

  if (!this->m_conv)
    throw std::runtime_error("Initial IPF failed to converge, check seed and marginals");

  doit(seed);
}

void QSIPF::doit(const NDArray<double>& seed)
{
  // Sample without replacement of static IPF
  // the original population (IPF::m_population will reduce as we sample)
  m_sample.assign(0);
  Sobol qrng(seed.dim());
  //const double scale = 0.5 / (1u<<31);
  // TODO use Index
  Index index(seed.sizes());
  for (size_t i = 0; i < m_originalPopulation; ++i)
  {
    if (!this->m_conv)
      throw std::runtime_error("IPF convergence failure");
    //const std::vector<uint32_t>& r = qrng.buf();

    getIndex(this->m_result, qrng.buf(), index);

    for (size_t d = 0; d < seed.dim(); ++d)
      if (index[d] >= this->m_result.sizes()[d])
        throw std::runtime_error("index out of bounds at " + std::to_string(d) + ": " + std::to_string(index[d]));

    // without replacement
    // directly decrementing the IPF population doesnt converge
    decrementMarginals(this->m_marginals, index);

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

const NDArray<int64_t>& QSIPF::sample() const
{
  return m_sample;
}

size_t QSIPF::population() const
{
  return m_originalPopulation;
}

// This returns the number of times the IPF population was recalculated
size_t QSIPF::iters() const
{
  return m_recalcs;
}

// chi-squared stat vs the IPF solution
double QSIPF::chiSq() const 
{
  double chisq = 0.0;
  for (Index index(m_sample.sizes()); !index.end(); ++index)
  {
    // m is the mean population of this state
    chisq += (m_sample[index] - m_ipfSolution[index]) * (m_sample[index] - m_ipfSolution[index]) / m_ipfSolution[index];
  }
  return chisq;
}
