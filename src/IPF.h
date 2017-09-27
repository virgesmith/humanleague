// IPF.h
// C++ implementation of multidimensional* iterative proportional fitting
// marginals are 1d in this implementation

#pragma once

#include "NDArray.h"
#include "Microsynthesis.h"

#include <vector>

class IPF
{
public:

  // construct from fractional marginals
  IPF(const NDArray<double>& seed, const std::vector<std::vector<double>>& marginals);

  // construct from integer marginals
  IPF(const NDArray<double>& seed, const std::vector<std::vector<int64_t>>& marginals);

  IPF(const IPF&) = delete;
  IPF(IPF&&) = delete;

  IPF& operator=(const IPF&) = delete;
  IPF& operator=(IPF&&) = delete;

  virtual ~IPF() { }

  void solve(const NDArray<double>& seed);

  virtual size_t population() const;

  const NDArray<double>& result() const;

  const std::vector<std::vector<double>> errors() const;

  double maxError() const;

  virtual bool conv() const;

  virtual size_t iters() const;

protected:

  bool computeErrors(std::vector<std::vector<double>>& diffs);

  static const size_t s_MAXITER = 10;

  NDArray<double> m_result;
  std::vector<std::vector<double>> m_marginals;
  std::vector<std::vector<double>> m_errors;
  size_t m_population;
  size_t m_iters;
  bool m_conv;
  const double m_tol = 1e-8;
  double m_maxError;
};

namespace wip {

class IPF : public Microsynthesis<double>
{
public:
  // TODO perhaps seed should be an arg to solve instead of being passed in here
  IPF(/*const NDArray<double>& seed,*/ const index_list_t& indices, marginal_list_t& marginals);

  NDArray<double>& solve();

  bool computeErrors(std::vector<NDArray<double>>& diffs);

  void rScale();
  
  void rDiff(std::vector<NDArray<double>>& diffs, const NDArray<double>& result, const std::vector<NDArray<double>>& marginals);

private:
  NDArray<double> m_seed;

  size_t m_iters;
  bool m_conv;
  double m_maxError;
  const double m_tol = 1e-8;

};

} // wip
