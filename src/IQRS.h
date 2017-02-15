
#pragma once

#include "NDArray.h"
#include "NDArrayUtils.h"
#include "DDWR.h"
#include "Sobol.h"
#include "PValue.h"

// n-Dimensional Integer Quasirandom with-Replacement Sampling
template<size_t D>
class IQRS
{
public:

  static const size_t Dim = D;

  typedef NDArray<Dim, uint32_t> table_t;

  typedef std::vector<uint32_t> marginal_t;

  IQRS(const std::vector<marginal_t>& marginals) : m_marginals(marginals)
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

    // init sobol seq
    //m_sobol.skip(m_sum);
  }

  ~IQRS() { }

  bool solve(/*size_t maxAttempts = 4*/)
  {
    static Sobol sobol(Dim);
    sobol.skip(m_sum);

    // TODO make dists members?
    std::vector<discrete_distribution_with_replacement<uint32_t>> dists;
    for (size_t i = 0; i < Dim; ++i)
    {
      dists.push_back(discrete_distribution_with_replacement<uint32_t>(m_marginals[i].begin(), m_marginals[i].end()));
    }

    m_t.assign(0u);

    size_t idx[Dim];
    for (size_t i = 0; i < m_sum; ++i)
    {
      for (size_t i = 0; i < Dim; ++i)
      {
        idx[i] = dists[i](sobol);
      }
      ++m_t[idx];
    }

    std::vector<std::vector<int32_t>> r(Dim);
    calcResiduals<Dim>(r);

    // print(m_t.rawData(),m_t.storageSize(), m_t.sizes()[1]);
    // std::cout << m_sum << std::endl;

    // std::cout << "initial residuals" << std::endl;
    // for (size_t i = 0; i < Dim; ++i)
    // {
    //   print(r[i]);
    //   std::cout << std::accumulate(r[i].begin(), r[i].end(), 0) << std::endl;
    // }

    size_t m_attempts = 0;

    //while (!allZeros(r) && m_attempts < 1/*maxAttempts*/) not looping for now
    if (!allZeros(r))
    {
      adjust3<Dim>(r); // is is adjusted on the fly
      // std::cout << "adjusted residuals" << std::endl;
      // for (size_t i = 0; i < Dim; ++i)
      // {
      //   print(r[i]);
      // }
      ++m_attempts;
    }

    for (size_t i = 0; i < Dim; ++i)
    {
      int32_t m = maxAbsElement(r[i]);
      m_residuals[i] = m;
      //allZero = allZero && (m == 0);
    }

    m_chi2 = 0.0;

    Index<D, Index_Unfixed> index(m_t.sizes());

    double scale = 1.0 / std::pow(m_sum, Dim-1u);

    while (!index.end())
    {
      // m is the mean population of this state
      double m = marginalProduct<Dim>(m_marginals, index) * scale;
      m_p[index] = m / m_sum;
      m_chi2 += (m_t[index] - m) * (m_t[index] - m) / m;
      ++index;
    }

    return allZeros(r);
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
  const NDArray<D, double>& stateProbabilities() const
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

  template<size_t O>
  void adjust3(std::vector<std::vector<int32_t>>& r)
  {
    adjust3<O-1>(r);
    //print(m_t.rawData(), m_t.storageSize());
    calcResiduals<Dim>(r);
    // recalc r
    adjust<Dim, O-1>(r, m_t, false);
    // TODO check we need the second call to calcResiduals
    calcResiduals<Dim>(r);
  }

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
  template<> \
  void IQRS<d>::calcResiduals<1>(std::vector<std::vector<int32_t>>& r) \
  { \
    r[0] = diff(reduce<d, uint32_t, 0>(m_t), m_marginals[0]); \
  }

#define SPECIALISE_ADJUST(d) \
  template<> \
  template<> \
  void IQRS<d>::adjust3<1>(std::vector<std::vector<int32_t>>& r) \
  { \
    adjust<d, 0>(r, m_t, false); \
    calcResiduals<1>(r); \
  }

SPECIALISE_CALCRESIDUALS(2)
SPECIALISE_CALCRESIDUALS(3)
SPECIALISE_CALCRESIDUALS(4)
SPECIALISE_CALCRESIDUALS(5)
SPECIALISE_CALCRESIDUALS(6)
SPECIALISE_CALCRESIDUALS(7)
SPECIALISE_CALCRESIDUALS(8)
SPECIALISE_CALCRESIDUALS(9)
SPECIALISE_CALCRESIDUALS(10)
SPECIALISE_CALCRESIDUALS(11)
SPECIALISE_CALCRESIDUALS(12)

SPECIALISE_ADJUST(2)
SPECIALISE_ADJUST(3)
SPECIALISE_ADJUST(4)
SPECIALISE_ADJUST(5)
SPECIALISE_ADJUST(6)
SPECIALISE_ADJUST(7)
SPECIALISE_ADJUST(8)
SPECIALISE_ADJUST(9)
SPECIALISE_ADJUST(10)
SPECIALISE_ADJUST(11)
SPECIALISE_ADJUST(12)

// remove the macros since this is a header file
#undef SPECIALISE_CALCRESIDUALS
#undef SPECIALISE_ADJUST


// Disallow nonsensical and trivial dimensionalities
template<> class IQRS<0>;
template<> class IQRS<1>;


