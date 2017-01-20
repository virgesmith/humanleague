
#pragma once


#include <algorithm>
#include <cstddef>
#include <cassert>

// this may need to be moved to a cpp file to avoid linker errors
static const size_t Index_Unfixed = -1ull;


// Indexer for elements in n-D array, holding dimension C constant
// Use C=Index_Unfixed to loop over all elements
template<size_t D, size_t C>
class Index
{
public:

  static const size_t Dim = D;

  Index(const size_t* sizes) : m_atEnd(false)
  {
    std::fill(m_idx, m_idx + Dim, 0);
    std::copy(sizes, sizes + D, m_sizes);
  }

  size_t* operator++()
  {
    for (size_t i = Dim - 1; i != -1ull; --i)
    {
      // ignore the iteration axis
      if (i == C) continue;

      ++m_idx[i];
      if (m_idx[i] != m_sizes[i])
        break;
      if (i == 0 || (C == 0 && i == 1))
        m_atEnd = true;
      m_idx[i] = 0;
    }
    return m_idx;
  }

  // implicit cast
  operator size_t*()
  {
    return &m_idx[0];
  }

  bool end()
  {
    return m_atEnd;
  }


private:
  size_t m_idx[Dim];
  size_t m_sizes[Dim];
  bool m_atEnd;
};

