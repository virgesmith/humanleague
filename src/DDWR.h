
#pragma once

#include <vector>
#include <algorithm>

template<typename I> // I must be an integral type
class discrete_distribution_with_replacement
{
public:

  typedef I value_type;

  // enforce integral types only
  static_assert(std::is_integral<I>::value, "discrete_distribution_with_replacement: only integral types supported");

  discrete_distribution_with_replacement(typename std::vector<I>::const_iterator b, typename std::vector<I>::const_iterator e)
  {
    m_freq.reserve(std::distance(b, e));
    std::copy(b, e, std::back_inserter(m_freq));
    m_sum = std::accumulate(m_freq.begin(), m_freq.end(), 0);
  }

  // std::distribution compatibility
  template<typename R>
  value_type operator()(R& rng)
  {
    return operator()(rng());
  }

  value_type operator()(value_type r)
  {
    r = r % m_sum;

    value_type idx = 0;
    value_type s = m_freq[0];
    while (r >= s)
    {
      ++idx;
      s += m_freq[idx];
    }
    return idx;
  }

private:
  std::vector<I> m_freq;
  I m_sum;
};

template<typename I> // I must be an integral type
class discrete_distribution_without_replacement
{
public:
  typedef I value_type;

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
  value_type operator()(R& rng)
  {
    return operator()(rng());
  }

  value_type operator()(value_type r)
  {
    if (!m_sum)
      throw std::runtime_error("distribution is depleted");

    r = r % m_sum;

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

  bool empty() const
  {
    return m_sum == 0;
  }

private:
  std::vector<I> m_freq;
  I m_sum;
};



