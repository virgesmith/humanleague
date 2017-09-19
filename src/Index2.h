
#pragma once


#include <algorithm>
#include <cstddef>
#include <cassert>

namespace wip {

// Indexer for elements in n-D array, optionally holding one dimension constant
class Index
{
public:
  static const int64_t Unfixed = -1;

  // Omit the second argument to loop over all elements
  explicit Index(const std::vector<int64_t>& sizes, const std::pair<int64_t, int64_t>& fixed = {-1, -1})
    : m_dim(sizes.size()), m_idx(sizes.size(), 0), m_sizes(sizes), m_fixed(fixed), m_atEnd(false)
  {
    assert(m_sizes.size());
    if (m_fixed.first != Unfixed)
    {
      m_idx[m_fixed.first] = m_fixed.second;
    }
    m_storageSize = m_sizes[0];
    for (size_t i = 1; i < m_dim; ++i)
      m_storageSize *= m_sizes[i];
  }

  Index(const Index& rhs)
    : m_dim(rhs.m_dim), m_idx(rhs.m_idx), m_sizes(rhs.m_sizes), m_fixed(rhs.m_fixed), m_storageSize(rhs.m_storageSize), m_atEnd(rhs.m_atEnd)
  {
  }

  const std::vector<int64_t>& operator++()
  {
    for (int64_t i = m_dim - 1; i != -1ll; --i)
    {
      // ignore the iteration axis
      if (i == m_fixed.first) continue;

      ++m_idx[i];
      if (m_idx[i] != m_sizes[i])
        break;
      if (i == 0 || (m_fixed.first == 0 && i == 1))
        m_atEnd = true;
      m_idx[i] = 0;
    }
    return m_idx;
  }

  // Implicitly cast to index vector
  operator const std::vector<int64_t>&() const
  {
    return m_idx;
  }

  size_t size() const 
  {
    return m_idx.size();
  }

  // allow read-only access to individual values
  const int64_t& operator[](size_t i) const
  {
    return m_idx[i];
  }

  // allow modification of individual values
  int64_t& operator[](size_t i)
  {
    return m_idx[i];
  }

  // NB row-major offset calc is in NDArray itself

  // need this for e.g. R where storage is column-major
  size_t colMajorOffset() const
  {
    size_t ret = 0;
    size_t mult = m_storageSize;
    for (int i = m_dim-1; i >= 0; --i)
    {
      mult /= m_sizes[i];
      ret += mult * m_idx[i];
    }
    return ret;
  }

  void reset()
  {
    m_idx.assign(m_dim, 0);
    m_atEnd = false;
  }

  bool end()
  {
    return m_atEnd;
  }

public:
  size_t m_dim;
  std::vector<int64_t> m_idx;
  std::vector<int64_t> m_sizes;
  // Fixed point (dim, idx)
  std::pair<int64_t, int64_t> m_fixed;
  size_t m_storageSize;
  bool m_atEnd;
};


// Contains a mapping from a higher dimensionality to a lower one
class MappedIndex
{
public:
  MappedIndex(Index& idx, const std::vector<int64_t>& mappedDimensions)
    : m_mappedIndex(mappedDimensions.size())
  {
    int64_t n = idx.size();
    (void)n; // avoid compiler warning about unused variable when assert exands to nothing
    // TODO check mappedDimensions are unique 
    for (size_t d = 0; d < m_mappedIndex.size(); ++d)
    {
      // check mappedDimensions are within dimension of index
      assert(mappedDimensions[d] < n);
      m_mappedIndex[d] = &idx[mappedDimensions[d]];
    }
  }

  // TODO better to overload NDArray to take Index types???
  operator const std::vector<int64_t*>&() const
  {
    return m_mappedIndex;
  }

private:
  std::vector<int64_t*> m_mappedIndex;
};

}
