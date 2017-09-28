
#include "QIS.h"

namespace wip {

QIS::QIS(/*const NDArray<double>& seed,*/ const index_list_t& indices, marginal_list_t& marginals)
: /*m_seed(std::move(seed)),*/ Microsynthesis(indices, marginals)
{

}

// Belongs in QIS?
const NDArray<double>& QIS::calcP()
{
  Index index_main(m_array.sizes());

  std::vector<MappedIndex> mappings = makeMarginalMappings(index_main);

  m_array.assign(1.0);
  for (; !index_main.end(); ++index_main)
  {
    for (size_t k = 0; k < m_marginals.size(); ++k)
    {
      m_array[index_main] *= m_marginals[k][mappings[k]];
    }
  }

  double scale = 1.0 / sum(m_array);
  for (Index index(m_array.sizes()); !index.end(); ++index)
  {
    m_array[index] *= scale;
  }

  return m_array;
}

}
