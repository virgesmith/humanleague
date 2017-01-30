
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
    std::copy(sizes, sizes + Dim, m_sizes);
    m_storageSize = m_sizes[0];
    for (size_t i = 1; i < Dim; ++i)
      m_storageSize *= m_sizes[i];
  }

  Index(const Index& rhs) : m_storageSize(rhs.m_storageSize), m_atEnd(rhs.m_atEnd)
  {
    std::copy(rhs.m_idx, rhs.m_idx + Dim, m_idx);
    std::copy(rhs.m_sizes, rhs.m_sizes + Dim, m_sizes);
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

  // NB row-major offset calc is in NDArray itself

  // need this for e.g. R where storage is column-major
  size_t colMajorOffset() const
  {
    size_t ret = 0;
    size_t mult = m_storageSize;
    for (int i = D-1; i >= 0; --i)
    {
      mult /= m_sizes[i];
      ret += mult * m_idx[i];
    }
    return ret;
  }

  bool end()
  {
    return m_atEnd;
  }

private:
  size_t m_idx[Dim];
  size_t m_sizes[Dim];
  size_t m_storageSize;
  bool m_atEnd;
};

// zero/one-d index unimplemented
template<size_t C> class Index<0, C>;
template<size_t C> class Index<1, C>;

