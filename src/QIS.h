#pragma once

#include "Microsynthesis.h"

namespace wip {

class QIS : public Microsynthesis<int64_t>
{
public:
  QIS(/*const NDArray<double>& seed,*/ const index_list_t& indices, marginal_list_t& marginals);

  const NDArray<int64_t>& solve();

  bool conv() const;

  // chi-squared stat vs the IPF solution
  double chiSq() const;

  // TODO p-value, degeneracy

private:

  void updateStateProbs();

  NDArray<double> m_stateProbs;
  // Required for chi-squared
  NDArray<double> m_expectedStateOccupancy;
  bool m_conv;
};

}
