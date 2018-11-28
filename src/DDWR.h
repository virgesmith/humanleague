
#pragma once

//#include "NDArray.h"

#include <vector>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <stdexcept>

template<typename T>
class discrete_distribution_with_replacement
{
public:

  typedef T value_type;
  typedef size_t index_type;
  typedef uint32_t random_type;

  // // enforce integral types only
  // static_assert(std::is_integral<I>::value, "discrete_distribution_with_replacement: only integral types supported");

  discrete_distribution_with_replacement(typename std::vector<T>::const_iterator b, typename std::vector<T>::const_iterator e)
  {
    m_freq.reserve(std::distance(b, e));
    std::copy(b, e, std::back_inserter(m_freq));
    m_sum = std::accumulate(m_freq.begin(), m_freq.end(), value_type());
  }

  // std::distribution compatibility
  template<typename R>
  size_t operator()(R& rng)
  {
    return operator()(rng());
  }

  index_type operator()(random_type r)
  {
    // map r [0,2^32-1] -> [0, m_sum) NB this effectively hard-codes random_type
    double p = (value_type)(double(r)/(1ull<<32) * m_sum);

    uint32_t idx = 0;
    value_type s = m_freq[0];
    while (p >= s)
    {
      ++idx;
      s += m_freq[idx];
    }
    return idx;
  }

  const value_type& operator[](size_t i) const
  {
    return m_freq[i];
  }

private:
  std::vector<T> m_freq;
  T m_sum;
};

template<typename I> // I must be an integral type
class discrete_distribution_without_replacement
{
public:
  typedef I result_type;

  static const result_type invalid_state = -1;

  // enforce integral types only
  static_assert(std::is_integral<I>::value, "discrete_distribution_without_replacement: only integral types supported");

  discrete_distribution_without_replacement(typename std::vector<I>::const_iterator b, typename std::vector<I>::const_iterator e)
  {
    m_freq.reserve(std::distance(b, e));
    std::copy(b, e, std::back_inserter(m_freq));
    m_sum = std::accumulate(m_freq.begin(), m_freq.end(), 0);
  }

  // std::distribution compatibility
  template<typename R>
  result_type operator()(R& rng)
  {
    return operator()(rng());
  }

  result_type operator()(result_type r)
  {
    if (!m_sum)
      throw std::runtime_error("distribution is depleted");

    // map r in [0,2^32) -> [0, m_sum)
    r = (uint32_t)(double(r)/(1ull<<32) * m_sum);

    I idx = 0;
    I s = m_freq[0];
    while (r >= s)
    {
      ++idx;
      s += m_freq[idx];
    }
    --m_freq[idx];
    --m_sum;
    return idx;
  }

  const std::vector<I>& freq() const { return m_freq; }

  result_type constrainedSample(result_type r, size_t firstForbiddenState/*const std::vector<uint32_t>& allowedStates*/)
  {
    if (!m_sum)
      throw std::runtime_error("distribution is depleted");

    firstForbiddenState = std::min(m_freq.size(), firstForbiddenState);

    result_type allowedSum = std::accumulate(m_freq.begin(), m_freq.begin() + firstForbiddenState, 0);

    // map r in [0,2^32) -> [0, m_sum)
    r = (uint32_t)(double(r)/(1ull<<32) * allowedSum);

    result_type idx = 0;
    result_type s = m_freq[0];
    while (r >= s)
    {
      ++idx;
      s += m_freq[idx];
      // give up if impossible to sample a non-forbidden state
      if (idx >= firstForbiddenState)
        return invalid_state;
    }
    --m_freq[idx];
    --m_sum;
    return idx;
}
  bool empty() const
  {
    return m_sum == 0;
  }

  // hack to allow us to modify the dist
  result_type& operator[](size_t i)
  {
    return m_freq[i];
  }

private:
  std::vector<I> m_freq;
  I m_sum;
};


// template<typename I> // I must be an integral type
// class constrained_distribution_without_replacement
// {
// public:
//   typedef I result_type;
//
//   static const result_type invalid_state = -1;
//
//   // enforce integral types only
//   static_assert(std::is_integral<I>::value, "discrete_distribution_without_replacement: only integral types supported");
//
//   constrained_distribution_without_replacement(const std::vector<I>& m0, const std::vector<I>& m1, const NDArray<2, bool>& permitted)
//     : m_d0(m0.begin(), m0.end()), m_constraint(std::move(permitted))
//   {
//     //m_d[0]
//     m_sum = std::accumulate(m0.begin(), m0.end(), 0);
//     if (m_sum != std::accumulate(m1.begin(), m1.end(), 0))
//     {
//       throw std::runtime_error("marginal total mismatch");
//     }
//   }
//
//   // std::distribution compatibility
//   template<typename R>
//   result_type operator()(R& rng)
//   {
//     return operator()(rng());
//   }
//
//   result_type operator()(result_type r)
//   {
//     NDArray<2, double> p(m_constraint.sizes());
//     p.assign(0);
//     size_t idx[2];
//
//     // calculate probabilities and zero forbidden states
//     for (idx[0] = 0; idx[0] < m_d0.size(); ++idx[0])
//     {
//       for (1[idx] = 0; idx[1] < m_d1.size(); ++idx[1])
//       {
//         p[idx] = m_constraint[idx] * double(m_d0[idx[0]] * m_d1[idx[1]]) / (m_sum*m_sum);
//       }
//     }
//
//     // rescale probabilities
//     // NB different answers depending on how you do this, e.g by row, by col, by matrix
//     double rescale = 1.0 / std::accumulate(p.rawData(), p.rawData() + p.storageSize(), 0.0);
//     for (size_t i = 0; i < p.storageSize(); ++i)
//     {
//       p.rawData()[i] *= rescale;
//     }
//
//     // sample a point (flattening to 1d)
//     discrete_distribution_with_replacement<double> pdist(p.rawData(), p.rawData() + p.storageSize());
//
//
//
//     // if (!m_sum)
//     //   throw std::runtime_error("distribution is depleted");
//     //
//     // // map r in [0,2^32) -> [0, m_sum)
//     // r = (uint32_t)(double(r)/(1ull<<32) * m_sum);
//
//     I idx1 = pdist(r);
//     // I s = m_freq[0];
//     // while (r >= s)
//     // {
//     //   ++idx;
//     //   s += m_freq[idx];
//     // }
//     // --m_freq[idx];
//     // --m_sum;
//     return idx1;
//   }
//
//   // result_type constrainedSample(result_type r, size_t firstForbiddenState/*const std::vector<uint32_t>& allowedStates*/)
//   // {
//   //   if (!m_sum)
//   //     throw std::runtime_error("distribution is depleted");
//   //
//   //   firstForbiddenState = std::min(m_freq.size(), firstForbiddenState);
//   //
//   //   result_type allowedSum = std::accumulate(m_freq.begin(), m_freq.begin() + firstForbiddenState, 0);
//   //
//   //   // map r in [0,2^32) -> [0, m_sum)
//   //   r = (uint32_t)(double(r)/(1ull<<32) * allowedSum);
//   //
//   //   result_type idx = 0;
//   //   result_type s = m_freq[0];
//   //   while (r >= s)
//   //   {
//   //     ++idx;
//   //     s += m_freq[idx];
//   //     // give up if impossible to sample a non-forbidden state
//   //     if (idx >= firstForbiddenState)
//   //       return invalid_state;
//   //   }
//   //   --m_freq[idx];
//   //   --m_sum;
//   //   return idx;
//   // }
//
//   bool empty() const
//   {
//     return m_sum == 0;
//   }
//
// private:
//   discrete_distribution_without_replacement<I> m_d0;
//   discrete_distribution_without_replacement<I> m_d1;
//
//   NDArray<2, bool> m_constraint;
//   I m_sum;
// };



