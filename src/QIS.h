#pragma once

#include "Microsynthesis.h"

namespace wip {

class QIS : public Microsynthesis<int64_t>
{
public:
  QIS(/*const NDArray<double>& seed,*/ const index_list_t& indices, marginal_list_t& marginals);

  const NDArray<int64_t>& solve();

private:

  void updateStateProbs();

  NDArray<double> m_stateProbs;
};

}
