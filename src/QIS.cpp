
#include "QIS.h"
#include "Index.h"
#include "Sobol.h"

namespace wip {

QIS::QIS(/*const NDArray<double>& seed,*/ const index_list_t& indices, marginal_list_t& marginals)
: /*m_seed(std::move(seed)),*/ Microsynthesis(indices, marginals)
{
  m_stateProbs.resize(m_array.sizes());
}

// control state of Sobol via arg?
const NDArray<int64_t>& QIS::solve()
{
  Index main_index(m_array.sizes());
  const std::vector<MappedIndex>& mappedIndices = makeMarginalMappings(main_index);
  m_array.assign(0ll);

  Sobol sobol_seq(m_dim);
  for (int64_t i = 0; i < m_population; ++i)
  {
    // calculate state probabilities
    updateStateProbs();

    // map sobol to a point in state space, store in index
    const std::vector<uint32_t>& seq = sobol_seq.buf();
    // ...
    (void)seq;

    // increment population
    ++m_array[main_index];

    // decrement marginals
    for (size_t j = 0; j < mappedIndices.size(); ++j)
    {
      --m_marginals[j][mappedIndices[j]];
    }
  }
  return m_array;
}

//
void QIS::updateStateProbs()
{
  Index index_main(m_array.sizes());

  std::vector<MappedIndex> mappings = makeMarginalMappings(index_main);

  m_stateProbs.assign(1.0);
  for (; !index_main.end(); ++index_main)
  {
    for (size_t k = 0; k < m_marginals.size(); ++k)
    {
      m_stateProbs[index_main] *= m_marginals[k][mappings[k]];
    }
  }

  // TODO rescaling is unnecessary
  double scale = 1.0 / sum(m_stateProbs);
  for (Index index(m_array.sizes()); !index.end(); ++index)
  {
    m_stateProbs[index] *= scale;
  }
}

} //wip