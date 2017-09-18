
#pragma once

#include "NDArray2.h"
#include "NDArrayUtils.h"
#include "NDArrayUtils2.h"
#include "Index2.h"
#include "Sobol.h"
#include "DDWR.h"
#include "StatFuncs.h"
#include <stdexcept>
#include <cmath>

// n-Dimensional without-replacement sampling
class QIWS
{
public:

  typedef wip::NDArray<uint32_t> table_t;

  typedef std::vector<uint32_t> marginal_t;

  explicit QIWS(const std::vector<marginal_t>& marginals) : m_dim(marginals.size()), m_marginals(marginals), m_residuals(marginals.size())
  {
    if (m_dim < 2)
      throw std::runtime_error("invalid dimension, must be > 1");
    // check for -ve values have to loop manually and convert to signed value :(
    for (size_t i = 0; i < m_marginals.size(); ++i)
    {
      for (size_t j = 0; j < m_marginals[i].size(); ++j)
      {
        if (static_cast<int32_t>(m_marginals[i][j]) < 0)
          throw std::runtime_error("negative marginal value in marginal " + std::to_string(i) + " element " + std::to_string(j));
      }
    }

    std::vector<int64_t> sizes(m_dim);
    m_sum = sum(m_marginals[0]);
    sizes[0] = m_marginals[0].size();
    m_dof = sizes[0] - 1;
    for (size_t i = 1; i < m_marginals.size(); ++i)
    {
      if (m_sum != sum(m_marginals[i]))
      {
        throw std::runtime_error("invalid marginals");
      }
      sizes[i] = m_marginals[i].size();
      m_dof *= sizes[i] - 1;
    }
    m_t.resize(sizes);
    m_p.resize(sizes);

    // keep lint happy
    m_chi2 = 0.0;
    m_degeneracy = 0.0;
  }

  virtual ~QIWS() { }

  bool solve()
  {
    // only initialised on first call, ensures different population each time
    // will throw when it reaches 2^32 samples
    static Sobol sobol(m_dim, m_sum);
    //static std::mt19937 sobol(70858048);

    std::vector<discrete_distribution_without_replacement<uint32_t>> dists;
    for (size_t i = 0; i < m_dim; ++i)
    {
      dists.push_back(discrete_distribution_without_replacement<uint32_t>(m_marginals[i].begin(), m_marginals[i].end()));
    }

    m_t.assign(0u);

    wip::Index idx(m_t.sizes());
    for (size_t j = 0; j < m_sum; ++j)
    {
      for (size_t i = 0; i < m_dim; ++i)
      {
        idx[i] = dists[i](sobol);
      }
      //print(idx, m_dim);
      ++m_t[idx];
    }

    std::vector<std::vector<int32_t>> r(m_dim);
    calcResiduals(r);

    bool allZero = true;
    for (size_t i = 0; i < m_dim; ++i)
    {
      int32_t m = maxAbsElement(r[i]);
      m_residuals[i] = m;
      allZero = allZero && (m == 0);
    }

    double scale = 1.0 / std::pow(m_sum, m_dim-1);

    for (wip::Index index(m_t.sizes()); !index.end(); ++index)
    {
      // m is the mean population of this state
      double m = marginalProduct<uint32_t>(m_marginals, index) * scale;
      m_p[index] = m / m_sum;
      m_chi2 += (m_t[index] - m) * (m_t[index] - m) / m;
    }

    return allZero;
  }

  std::pair<double, bool> pValue() const
  {
    return ::pValue(m_dof, m_chi2);
  }

  double chiSq() const
  {
    return m_chi2;
  }

  const table_t& result() const
  {
    return m_t;
  }

  const std::vector<int32_t>& residuals() const
  {
    return m_residuals;
  }

  size_t population() const
  {
    return m_sum;
  }

  // the mean population of each state
  const wip::NDArray<double>& stateProbabilities() const
  {
    return m_p;
  }

protected:

  void calcResiduals(std::vector<std::vector<int32_t>>& r)
  {
    for (size_t d = 0; d < r.size(); ++d)
    {
      r[d] = diff(reduce<uint32_t>(m_t, d), m_marginals[d]);
    }
  }

  size_t m_dim;
  const std::vector<marginal_t> m_marginals;
  table_t m_t;
  // probabilities for each state
  wip::NDArray<double> m_p;
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

// // TODO helper macro for member template specialisations
// #define SPECIALISE_CALCRESIDUALS(d) \
//   template<> \
//   template<> \
//   inline void QIWS<d>::calcResiduals<1>(std::vector<std::vector<int32_t>>& r) \
//   { \
//     r[0] = diff(reduce<d, uint32_t, 0>(m_t), m_marginals[0]); \
//   }
