#pragma once

#include "Microsynthesis.h"
#include "Sobol.h"

class QIS : public Microsynthesis<int64_t>
{
public:
  QIS(const index_list_t& indices, marginal_list_t& marginals, int64_t skips = 0);

  // TODO need a mechanism to invalidate result after it's been moved (or just copy it)
  const NDArray<int64_t>& solve(bool reset = false);
  //const NDArray<int64_t>& solve2();

  bool conv() const;

  // chi-squared stat vs the IPF solution
  double chiSq() const;

  double degeneracy() const;

  double pValue() const;

private:

  void updateStateProbs();

  Sobol m_sobolSeq;

  NDArray<double> m_stateProbs;
  // Required for chi-squared
  NDArray<double> m_expectedStateOccupancy;
  double m_chiSq;
  double m_pValue;
  double m_degeneracy;
  bool m_conv;
};

