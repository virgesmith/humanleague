
#pragma once

#include "NDArray.h"
#include "NDArrayUtils.h"
#include "Sobol.h"
#include "DDWR.h"

// inline bool allZeros(const std::vector<std::vector<int32_t>>& r)
// {
//   for (const auto& v: r)
//     if (!isZero(v))
//       return false;
//   return true;
// }

inline int32_t maxAbsElement(const std::vector<int32_t>& r)
{
  int32_t m = 0;
  for (size_t i = 0; i < r.size(); ++i)
  {
    m = std::max(m, abs(r[i]));
  }
  return m;
}

inline std::vector<int32_t> diff(const std::vector<uint32_t>& x, const std::vector<uint32_t>& y)
{
  size_t size = x.size();
  assert(size == y.size());

  std::vector<int32_t> result(size);

  for (size_t i = 0; i < x.size(); ++i)
  {
    result[i] = x[i] - y[i];
  }
  return result;
}



// n-Dimensional Quasirandom integer proportional(?) fitting
template<size_t D>
class QIPF
{
public:

  static const size_t Dim = D;

  typedef NDArray<Dim, uint32_t> table_t;

  typedef std::vector<uint32_t> marginal_t;

  QIPF(const std::vector<marginal_t>& marginals) : m_marginals(marginals)//, m_attempts(0ull), m_sobol(Dim)
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

  bool solve()
  {
    // only initialised on first call, ensures different population each time
    static Sobol sobol(Dim, m_sum);
    //static std::mt19937 sobol(70858048);

    std::vector<discrete_distribution_without_replacement<uint32_t>> dists;
    for (size_t i = 0; i < Dim; ++i)
    {
      dists.push_back(discrete_distribution_without_replacement<uint32_t>(m_marginals[i].begin(), m_marginals[i].end()));
    }

    m_t.assign(0u);

    size_t idx[Dim];
    for (size_t j = 0; j < m_sum; ++j)
    {
      for (size_t i = 0; i < Dim; ++i)
      {
        idx[i] = dists[i](sobol());
      }
      //print(idx, Dim);
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

    return allZero;
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

  const int32_t* residuals() const
  {
    return m_residuals;
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
  size_t m_sum;
  int32_t m_residuals[Dim];
  //Sobol m_sobol;
};

// TODO helper macro for member template specialisations
#define SPECIALISE_CALCRESIDUALS(d) \
  template<> \
  template<> \
  inline void QIPF<d>::calcResiduals<1>(std::vector<std::vector<int32_t>>& r) \
  { \
    r[0] = diff(reduce<d, uint32_t, 0>(m_t), m_marginals[0]); \
  }

#define SPECIALISE_ADJUST(d) \
  template<> \
  template<> \
  inline void QIPF<d>::adjust3<1>(std::vector<std::vector<int32_t>>& r) \
  { \
    adjust<d, 0>(r[0], m_t, true); \
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

// SPECIALISE_ADJUST(2)
// SPECIALISE_ADJUST(3)
// SPECIALISE_ADJUST(4)
// SPECIALISE_ADJUST(5)
// SPECIALISE_ADJUST(6)
// SPECIALISE_ADJUST(7)
// SPECIALISE_ADJUST(8)
// SPECIALISE_ADJUST(9)
// SPECIALISE_ADJUST(10)
// SPECIALISE_ADJUST(11)
// SPECIALISE_ADJUST(12)

// Disallow nonsensical and trivial dimensionalities
template<> class QIPF<0>;
template<> class QIPF<1>;




