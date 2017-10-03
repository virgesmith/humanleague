#pragma once

#include "Microsynthesis.h"
#include "Sobol.h"

class QISI : public Microsynthesis<int64_t>
{
public:
  QISI(const index_list_t& indices, marginal_list_t& marginals, int64_t skips = 0);

  // TODO need a mechanism to invalidate result after it's been moved (or just copy it)
  const NDArray<int64_t>& solve(const NDArray<double>& seed, bool reset = false);

  bool conv() const;

  // chi-squared stat vs the IPF solution
  double chiSq() const;

  double degeneracy() const;

  double pValue() const;

private:

  void recomputeIPF(const NDArray<double>& seed);

  Sobol m_sobolSeq;
  NDArray<double> m_expectedStateOccupancy;
  // Required for chi-squared
  NDArray<double> m_ipfSolution;
  double m_chiSq;
  double m_pValue;
  double m_degeneracy;
  bool m_conv;
};

