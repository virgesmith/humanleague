#pragma once

#include "Microsynthesis.h"

namespace wip {

class QIS : public Microsynthesis<int64_t>
{
public:
  QIS(const index_list_t& indices, marginal_list_t& marginals);

  // TODO need a mechanism to invalidate result after it's been moved (or just copy it)
  const NDArray<int64_t>& solveFast();
  const NDArray<int64_t>& solve();
  
  bool conv() const;

  // chi-squared stat vs the IPF solution
  double chiSq() const;

  double degeneracy() const;

  double pValue() const;

private:

  void updateStateProbs();

  NDArray<double> m_stateProbs;
  // Required for chi-squared
  NDArray<double> m_expectedStateOccupancy;
  double m_chiSq;
  double m_pValue;
  double m_degeneracy;
  bool m_conv;
};

}
