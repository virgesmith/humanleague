
#pragma once

#include "NDArray.h"
#include "NDArrayUtils.h"
#include "Sobol.h"
#include "DDWR.h"
#include "PValue.h"
#include <cmath>


// 2-Dimensional constrained quasirandom integer without-replacement sampling
// constraint is hard-coded (for now) to: idx1 <= idx0
// TODO rename
//template<size_t D>
class CQIWS
{
public:

  static const size_t Dim = 2;

  typedef NDArray<Dim, uint32_t> table_t;

  typedef std::vector<uint32_t> marginal_t;

  CQIWS(const std::vector<marginal_t>& marginals) : m_marginals(marginals)
  {
    if (m_marginals.size() != Dim)
    {
      throw std::runtime_error("invalid no. of marginals");
    }

    // check for -ve values have to loop manually and convert to signed value :(
    for (size_t i = 0; i < m_marginals.size(); ++i)
    {
      for (size_t j = 0; j < m_marginals[i].size(); ++j)
      {
        if (static_cast<int32_t>(m_marginals[i][j]) < 0)
          throw std::runtime_error("negative marginal value in marginal " + std::to_string(i) + " element " + std::to_string(j));
      }
    }

    size_t sizes[Dim];
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
    m_t.resize(&sizes[0]);
    m_p.resize(&sizes[0]);
  }

  ~CQIWS() { }

  bool solve()
  {
    // only initialised on first call, ensures different population each time
    // will throw when it reaches 2^32 samples
    static Sobol sobol(Dim, m_sum);
    //static std::mt19937 sobol(70858048);

    std::vector<discrete_distribution_without_replacement<uint32_t>> dists;
    for (size_t i = 0; i < Dim; ++i)
    {
      dists.push_back(discrete_distribution_without_replacement<uint32_t>(m_marginals[i].begin(), m_marginals[i].end()));
    }

    m_t.assign(0u);

    size_t idx[Dim];

    for (uint32_t i = 0; i < m_sum; ++i)
    {
      idx[0] = dists[0](sobol);

      size_t constraint = idx[0] + 1;

      idx[1] = dists[1].constrainedSample(sobol(), constraint);
      if (idx[1] == discrete_distribution_without_replacement<uint32_t>::invalid_state)
        break;

      ++m_t[idx];
    }

    std::vector<std::vector<int32_t>> r(Dim);
    calcResiduals<Dim>(r);

    bool allZero = true;
    for (size_t i = 0; i < Dim; ++i)
    {
      int32_t m = maxAbsElement(r[i]);
      m_residuals[i] = m;
      allZero = allZero && (m == 0);
    }

    m_chi2 = 0.0;

    Index<Dim, Index_Unfixed> index(m_t.sizes());

    double scale = 1.0 / std::pow(m_sum, Dim-1);

    while (!index.end())
    {
      // m is the mean population of this state
      double m = marginalProduct<Dim>(m_marginals, index) * scale;
      m_p[index] = m / m_sum;
      m_chi2 += (m_t[index] - m) * (m_t[index] - m) / m;
      ++index;
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

  const int32_t* residuals() const
  {
    return m_residuals;
  }

  size_t population() const
  {
    return m_sum;
  }

  // the mean population of each state
  const NDArray<Dim, double>& stateProbabilities() const
  {
    return m_p;
  }

private:

  template<size_t O>
  void calcResiduals(std::vector<std::vector<int32_t>>& r)
  {
    calcResiduals<O-1>(r);
    r[O-1] = diff(reduce<Dim, uint32_t, O-1>(m_t), m_marginals[O-1]);
  }

  // template<size_t O>
  // void adjust3(std::vector<std::vector<int32_t>>& r)
  // {
  //   adjust3<O-1>(r);
  //   //print(m_t.rawData(), m_t.storageSize());
  //   calcResiduals<Dim>(r);
  //   // recalc r
  //   adjust<Dim, O-1>(r[O-1], m_t, true);
  //   // TODO check we need the second call to calcResiduals
  //   calcResiduals<Dim>(r);
  // }

  const std::vector<marginal_t> m_marginals;
  table_t m_t;
  // probabilities for each state
  NDArray<Dim, double> m_p;
  // total population
  size_t m_sum;
  // difference between table sums (over single dim) and marginal
  int32_t m_residuals[Dim];
  // chi-squared statistic
  double m_chi2;
  // degrees of freedom (for p-value calculation)
  uint32_t m_dof;
};

// TODO helper macro for member template specialisations
#define SPECIALISE_CALCRESIDUALS(d) \
  template<> \
  inline void CQIWS::calcResiduals<1>(std::vector<std::vector<int32_t>>& r) \
  { \
    r[0] = diff(reduce<d, uint32_t, 0>(m_t), m_marginals[0]); \
  }

SPECIALISE_CALCRESIDUALS(2)


// remove the macros since this is a header file
#undef SPECIALISE_CALCRESIDUALS





