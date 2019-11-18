
#include "QISI.h"
#include "IPF.h"
#include "Index.h"
#include "StatFuncs.h"
#include "Log.h"

namespace {

// TODO move somewhere appropriate (doesnt need to be member) (copy&paste from QSIPF)
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
  throw std::runtime_error("pick failed: %% from %%"_s % r % dist);
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


QISI::QISI(const index_list_t& indices, marginal_list_t& marginals, int64_t skips)
: Microsynthesis(indices, marginals), m_sobolSeq(m_dim), 
  m_chiSq(std::numeric_limits<double>::quiet_NaN()), 
  m_pValue(std::numeric_limits<double>::quiet_NaN()), 
  m_degeneracy(std::numeric_limits<double>::quiet_NaN()),
  m_conv(false)
{
  m_sobolSeq.skip(skips);
}

// control state of Sobol via arg?
const NDArray<int64_t>& QISI::solve(const NDArray<double>& seed, bool reset)
{
  // check seed dimensions consistent with marginals
  if (seed.dim() != m_array.dim())
    throw std::runtime_error("seed dimensions %% is inconsistent with that implied by marginals (%%)"_s % seed.dim() % m_array.dim());
  for (size_t d = 0; d < m_array.dim(); ++d)
  {
    if (seed.sizes()[d] != m_array.sizes()[d])
      throw std::runtime_error("seed dimensions %% are inconsistent with that implied by marginals (%%)"_s % seed.sizes() % m_array.sizes());
  }

  if (reset)
  {
    m_sobolSeq.reset();
  }

  m_ipfSolution.resize(m_array.sizes());
  m_expectedStateOccupancy.resize(m_array.sizes());
  // compute initial IPF solution and keep a copy
  recomputeIPF(seed);
  NDArray<double>::copy(m_ipfSolution, m_expectedStateOccupancy);

  m_conv = true;
  Index main_index(m_array.sizes());
  const std::vector<MappedIndex>& mappedIndices = makeMarginalMappings(main_index);
  m_array.assign(0ll);

  Sobol sobol_seq(m_dim);
  for (int64_t i = 0; i < m_population; ++i)
  {
    // map sobol to a point in state space, store in index
    const std::vector<uint32_t>& seq = sobol_seq.buf();
    // ...
    getIndex(m_ipfSolution, seq, main_index);

    //print((std::vector<int64_t>)main_index);
    //print(m_ipfSolution.rawData(), m_ipfSolution.storageSize());
    // increment population
    ++m_array[main_index];

    // decrement marginals, checking none have gone -ve
    for (size_t j = 0; j < mappedIndices.size(); ++j)
    {
      --m_marginals[j][mappedIndices[j]];
      if (m_marginals[j][mappedIndices[j]] < 0)
        m_conv = false;
    }

    // recalculate state probabilities only do when state occupancy drops below zero
#ifdef VERBOSE
    print(m_ipfSolution.rawData(), m_ipfSolution.storageSize(), m_ipfSolution.sizes()[0]);
#endif
    --m_ipfSolution[main_index];
    if (m_ipfSolution[main_index] < 0.0)
      recomputeIPF(seed);
  }
  m_chiSq = ::chiSq(m_array, m_expectedStateOccupancy);

  m_pValue = ::pValue(dof(m_array.sizes()), m_chiSq).first;

  m_degeneracy = ::degeneracy(m_array);

  return m_array;
}

// Expected state occupancy
const NDArray<double>& QISI::expectation()
{
  return m_expectedStateOccupancy;
}


//
void QISI::recomputeIPF(const NDArray<double>& seed)
{
  // TODO make more efficient
  // is this moving the marginals???
  IPF<int64_t> ipf(m_indices, m_marginals);
  NDArray<double>::copy(ipf.solve(seed), m_ipfSolution);
}

double QISI::chiSq() const
{
  return m_chiSq;
}

double QISI::pValue() const
{
  return m_pValue;
}

double QISI::degeneracy() const
{
  return m_degeneracy;
}

bool QISI::conv() const
{
  return m_conv;
}

