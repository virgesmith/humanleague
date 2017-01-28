
#pragma once

#include "NDArray.h"
#include "NDArrayUtils.h"
#include "Sobol.h"


// n-Dimensional Quasirandom integer proportional(?) fitting
template<size_t D>
class QIPF
{
public:

  static const size_t Dim = D;

  typedef NDArray<Dim, uint32_t> table_t;

  typedef std::vector<uint32_t> marginal_t;

  QIPF(const std::vector<marginal_t>& marginals) : m_marginals(marginals), m_attempts(0ull)
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
    for (size_t i = 1; i < m_marginals.size(); ++i)
    {
      if (m_sum != sum(m_marginals[i]))
      {
        throw std::runtime_error("invalid marginals");
      }
      sizes[i] = m_marginals[i].size();
    }
    m_t.resize(&sizes[0]);
  }

  ~QIPF() { }

  bool solve(size_t maxAttempts = 4)
  {
    /*static*/ Sobol sobol(Dim);

    std::vector<std::discrete_distribution<uint32_t>> dists;
    for (size_t i = 0; i < Dim; ++i)
    {
      dists.push_back(std::discrete_distribution<uint32_t>(m_marginals[i].begin(), m_marginals[i].end()));
    }

    bool success = false;

    for (m_attempts = 0; m_attempts < maxAttempts && !success; ++m_attempts)
    {
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

      // print(m_t.rawData(),16);

      // std::cout << "initial residuals" << std::endl;
      // for (size_t i = 0; i < Dim; ++i)
      // {
      //  print(r[i]);
      // }

      success = adjust2<Dim>(r);
      calcResiduals<Dim>(r);

      // std::cout << "adjusted residuals" << std::endl;
      // for (size_t i = 0; i < Dim; ++i)
      // {
      //   print(r[i]);
      // }
      //std::cout << "sample = " << sample << std::endl;
      //std::cout << "success = " << success << std::endl;

    }

    return success;
  }

  double msv() const
  {
    double sumSq = 0.0;

    Index<D, Index_Unfixed> idx(m_t.sizes());

    double scale = 1.0 / std::pow(m_sum, Dim-1);

    while (!idx.end())
    {
      double f = marginalProduct<Dim>(m_marginals, idx) * scale;
      sumSq += (f - m_t[idx]) * (f - m_t[idx]);
      ++idx;
    }

    return sumSq / m_t.storageSize();
  }

  const table_t& result() const
  {
    return m_t;
  }

  size_t population() const
  {
    return m_sum;
  }

  // the mean population of each state
  double meanPopPerState() const
  {
    return double(m_sum) / m_t.storageSize();
  }

  size_t attempts() const
  {
    return m_attempts;
  }

private:

  template<size_t O>
  void calcResiduals(std::vector<std::vector<int32_t>>& r)
  {
    calcResiduals<O-1>(r);
    r[O-1] = diff(reduce<Dim, uint32_t, O-1>(m_t), m_marginals[O-1]);
  }

  template<size_t O>
  bool adjust2(const std::vector<std::vector<int32_t>>& r)
  {
    bool ret = true;
    ret = ret && adjust2<O-1>(r);
    ret = ret && adjust<Dim, O-1>(r[O-1], m_t);
    return ret;
  }

  const std::vector<marginal_t> m_marginals;
  table_t m_t;
  size_t m_sum;
  size_t m_attempts;
};

// TODO helper macro for member template specialisations
#define SPECIALISE_CALCRESIDUALS(d) \
  template<> \
  template<> \
  void QIPF<d>::calcResiduals<1>(std::vector<std::vector<int32_t>>& r) \
  { \
    r[0] = diff(reduce<d, uint32_t, 0>(m_t), m_marginals[0]); \
  }

#define SPECIALISE_ADJUST(d) \
  template<> \
  template<> \
  bool QIPF<d>::adjust2<1>(const std::vector<std::vector<int32_t>>& r) \
  { \
    return adjust<d, 0>(r[0], m_t); \
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

// Disallow nonsensical and trivial dimensionalities
template<> class QIPF<0>;
template<> class QIPF<1>;




