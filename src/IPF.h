// IPF.h
// C++ implementation of multidimensional* iterative proportional fitting
// marginals are 1d in this implementation

#pragma once

#include "NDArray.h"
#include "Log.h"
#include "Microsynthesis.h"

#include <vector>
#include <limits>
#include <cmath>

template<typename M>
class IPF : public Microsynthesis<double, M> // marginal type
{
public:

  typedef typename Microsynthesis<double, M>::index_list_t index_list_t;
  typedef typename Microsynthesis<double, M>::marginal_list_t marginal_list_t;

  IPF(const typename Microsynthesis<double, M>::index_list_t& indices, typename Microsynthesis<double, M>::marginal_list_t& marginals)
    : Microsynthesis<double, M>(indices, marginals)
  {
  }

  IPF(const IPF&) = delete;
  IPF(IPF&&) = delete;

  IPF& operator=(const IPF&) = delete;
  IPF& operator=(IPF&&) = delete;

  ~IPF() { }

  // TODO need a mechanism to invalidate result after its been moved
  NDArray<double>& solve(const NDArray<double>& seed)
  {
    // check seed dimensions consistent with marginals
    if (seed.dim() != this->m_array.dim())
      throw std::runtime_error("seed dimensions %% is inconsistent with that implied by marginals (%%)"_s % seed.dim() % this->m_array.dim());
    for (size_t d = 0; d < this->m_array.dim(); ++d)
    {
      if (seed.sizes()[d] != this->m_array.sizes()[d])
        throw std::runtime_error("seed dimensions %% are inconsistent with that implied by marginals (%%)"_s % seed.sizes() % this->m_array.sizes());
    }

    Index index_main(this->m_array.sizes());

    std::vector<MappedIndex> mappings;
    mappings.reserve(this->m_marginals.size());
    for (size_t k = 0; k < this->m_marginals.size(); ++k)
    {
      mappings.push_back(MappedIndex(index_main, this->m_indices[k]));
    }

    //this->m_array.assign(1.0);
    std::copy(seed.rawData(), seed.rawData() + seed.storageSize(), const_cast<double*>(this->m_array.rawData()));

    std::vector<NDArray<double>> diffs(this->m_marginals.size());
    m_errors.resize(this->m_marginals.size());

    for (size_t k = 0; k < diffs.size(); ++k)
    {
      diffs[k].resize(this->m_marginals[k].sizes());
      m_errors[k].resize(this->m_marginals[k].sizes());
    }

    m_conv = false;
    for (m_iters = 0; !m_conv && m_iters < s_MAXITER; ++m_iters)
    {
      // move back into this class?
      Microsynthesis<double, M>::rScale();
      Microsynthesis<double, M>::rDiff(diffs);

      m_conv = computeErrors(diffs);
    }

    return this->m_array;
  }


  const std::vector<NDArray<double>>& errors() const
  {
    return m_errors;
  }

  double maxError() const
  {
    return m_maxError;
  }

  bool conv() const
  {
    return m_conv;
  }

  size_t iters() const
  {
    return m_iters;
  }

private:

  bool computeErrors(std::vector<NDArray<double>>& diffs)
  {
    m_maxError = -std::numeric_limits<double>::max();

    // // create mapped indices
    // const std::vector<MappedIndex>& mapped = makeMappings(main_index);
    for (size_t k = 0; k < diffs.size(); ++k)
    {
      for (Index index(diffs[k].sizes()); !index.end(); ++index)
      {
        double e = std::fabs(diffs[k][index]);
        m_errors[k][index] = e;
        m_maxError = std::max(m_maxError, e);
      }
    }

    return m_maxError < m_tol;
  }

  NDArray<double> m_seed;
  size_t m_iters;
  bool m_conv;
  Microsynthesis<double>::marginal_list_t m_errors;
  double m_maxError;
  const double m_tol = 1e-8;

  static const size_t s_MAXITER = 1000;
};

