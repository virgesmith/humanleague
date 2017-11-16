#pragma once

// Quasirandomly sampled IPF

#include "IPF.h"
#include "NDArray.h"

#include <vector>

class QSIPF : public deprecated::IPF
{
public:
  // TODO marginal values must be integers
  QSIPF(const NDArray<double>& seed, const std::vector<std::vector<int64_t>>& marginals);

  ~QSIPF() { }

  const NDArray<int64_t>& sample() const;

  virtual size_t population() const;

  // This returns the number of times the IPF population was recalculated
  virtual size_t iters() const;

  // chi-squared stat vs the IPF solution
  double chiSq() const ;

private:

  void doit(const NDArray<double>& seed);

  size_t m_originalPopulation;
  NDArray<int64_t> m_sample;
  NDArray<double> m_ipfSolution;
  size_t m_recalcs;
  //double m_chi2;
};
