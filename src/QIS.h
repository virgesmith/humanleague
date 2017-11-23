#pragma once

#include "Microsynthesis.h"
#include "Sobol.h"

class QIS : public Microsynthesis<int64_t>
{
public:
  QIS(const index_list_t& indices, marginal_list_t& marginals, int64_t skips = 0);

  // TODO need a mechanism to invalidate result after it's been moved (or just copy it)
  const NDArray<int64_t>& solve(bool reset = false);
  
  // Expected state occupancy
  const NDArray<double>& expectation();

  // convergence
  bool conv() const;

  // chi-squared stat vs the IPF solution
  double chiSq() const;

  double degeneracy() const;

  double pValue() const;

private:

  const NDArray<int64_t>& solve_p(bool reset);
  const NDArray<int64_t>& solve_m(bool reset);
  
  // state values are proportional to state occupancy probabilities
  void updateStateValues(const Index& position, const std::vector<MappedIndex>& mappings);
  void computeStateValues();

  Sobol m_sobolSeq;

  // values proportional to state probs
  NDArray<double> m_stateValues;
  // Required for chi-squared
  NDArray<double> m_expectedStateOccupancy;
  double m_chiSq;
  double m_pValue;
  double m_degeneracy;
  bool m_conv;
};

