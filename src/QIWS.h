
#pragma once

#include "NDArray.h"

// n-Dimensional without-replacement sampling
class QIWS
{
public:

  typedef NDArray<uint32_t> table_t;

  typedef std::vector<uint32_t> marginal_t;

  explicit QIWS(const std::vector<marginal_t>& marginals);

  virtual ~QIWS() { }

  bool solve();

  std::pair<double, bool> pValue() const;

  double chiSq() const;

  const table_t& result() const;

  const std::vector<int32_t>& residuals() const;

  size_t population() const;

  // the mean population of each state
  const NDArray<double>& stateProbabilities() const;

protected:

  void calcResiduals(std::vector<std::vector<int32_t>>& r);

  size_t m_dim;
  const std::vector<marginal_t> m_marginals;
  table_t m_t;
  // probabilities for each state
  NDArray<double> m_p;
  // total population
  size_t m_sum;
  // difference between table sums (over single dim) and marginal sum
  std::vector<int32_t> m_residuals;
  // chi-squared statistic
  double m_chi2;
  // degrees of freedom (for p-value calculation)
  uint32_t m_dof;
  // TODO degeneracy S!/Product_k(Tk!)
  double m_degeneracy;
};

