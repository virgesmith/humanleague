
#pragma once

#include "QISI.h"

#include <vector>
#include <memory>
#include <cstdlib>

// Given pop and real number densities (sum = 1), produce integer frequencies with minimal mean squuare error
std::vector<int> integeriseMarginalDistribution(const std::vector<double>& p, int pop, double& mse);

// Given a fractional population in n dimensions, with integral marginal sums, construct a QISI object using the 1d marginals
// Class wrapper around QISI for integerisation. Necessary as base class Microsynthesis only stores refs to the marginals for efficiency reasons
// This class needs to construct persistent marginals that can be safely referenced. But NB It only stores a ref of the seed array
class Integeriser final
{
public:
  Integeriser(const NDArray<double>& seed);

  ~Integeriser() = default;

  Integeriser(const Integeriser&) = delete;
  Integeriser& operator=(const Integeriser&) = delete;

  const NDArray<int64_t>& result() const;
  bool conv() const;
  double rmse() const;

private:
  const NDArray<double>& m_seed;
  QISI::index_list_t m_indices; 
  QISI::marginal_list_t m_marginals;

  NDArray<int64_t> m_result;
  bool m_conv;
  //double m_chi2;
  //double m_pvalue;
  double m_rmse;

};
