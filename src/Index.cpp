
#include "Index.h"

#include "NDArrayUtils.h"

#include <algorithm>
#include <cassert>


Index::Index(const std::vector<int64_t>& sizes)
  : m_dim(sizes.size()), m_idx(sizes.size(), 0), m_sizes(sizes), m_atEnd(false)
{
  assert(m_sizes.size());
  m_storageSize = m_sizes[0];
  for (size_t i = 1; i < m_dim; ++i)
    m_storageSize *= m_sizes[i];
}

const std::vector<int64_t>& Index::operator++()
{
  for (int64_t i = m_dim - 1; i != -1ll; --i)
  {
    ++m_idx[i];
    if (m_idx[i] != m_sizes[i])
      break;
    if (i == 0)
      m_atEnd = true;
    m_idx[i] = 0;
  }
  return m_idx;
}

// Implicitly cast to index vector
Index::operator const std::vector<int64_t>&() const
{
  return m_idx;
}

size_t Index::size() const
{
  return m_idx.size();
}

const std::vector<int64_t>& Index::sizes() const
{
  return m_sizes;
}


// allow read-only access to individual values
const int64_t& Index::operator[](size_t i) const
{
  return m_idx[i];
}

// allow modification of individual values
int64_t& Index::operator[](size_t i)
{
  return m_idx[i];
}

void Index::reset()
{
  m_idx.assign(m_dim, 0);
  m_atEnd = false;
}

bool Index::end() const
{
  return m_atEnd;
}


TransposedIndex::TransposedIndex(const std::vector<int64_t>& sizes)
: Index(sizes)
{ }

const std::vector<int64_t>& TransposedIndex::operator++()
{
  for (size_t i = 0; i < m_dim; ++i)
  {
    ++m_idx[i];
    if (m_idx[i] != m_sizes[i])
      break;
    if (i == m_dim-1)
      m_atEnd = true;
    m_idx[i] = 0;
  }
  return m_idx;
}


MappedIndex::MappedIndex(const Index& idx, const std::vector<int64_t>& mappedDimensions)
: m_dim(mappedDimensions.size()), m_sizes(m_dim), m_mappedIndex(m_dim), m_atEnd(idx.end())
{
  // TODO check mappedDimensions are unique
  for (size_t d = 0; d < m_dim; ++d)
  {
    // check mappedDimensions are within dimension of index
    assert(mappedDimensions[d] < idx.size());
    m_sizes[d] = idx.sizes()[mappedDimensions[d]];
    m_mappedIndex[d] = &const_cast<Index&>(idx)[mappedDimensions[d]];
  }
}

const MappedIndex& MappedIndex::operator++()
{
  for (int64_t i = m_dim - 1; i != -1ll; --i)
  {
    ++*m_mappedIndex[i];
    if (*m_mappedIndex[i] != m_sizes[i])
      break;
    if (i == 0)
      m_atEnd = true;
    *m_mappedIndex[i] = 0;
  }
  return *this;
}

// TODO better to overload NDArray to take Index types???
MappedIndex::operator const std::vector<int64_t*>&() const
{
  return m_mappedIndex;
}

// allow read-only access to individual values
const int64_t& MappedIndex::operator[](size_t i) const
{
  return *m_mappedIndex[i];
}

// allow modification of individual values
int64_t& MappedIndex::operator[](size_t i)
{
  return *m_mappedIndex[i];
}

bool MappedIndex::end()
{
  return m_atEnd;
}


FixedIndex::FixedIndex(const std::vector<int64_t>& sizes, const std::vector<std::pair<int64_t, int64_t>>& fixed)
  : m_freeDim(sizes.size() - fixed.size()), m_fullIndex(sizes), m_freeSizes(sizes.size() - fixed.size()), m_atEnd(false)
{
  // invalidate full index
  for (size_t i = 0; i < m_fullIndex.size(); ++i)
  {
    m_fullIndex[i] = -1;
  }

  // set fixed in full index
  for (size_t i = 0; i < fixed.size(); ++i)
  {
    m_fullIndex[fixed[i].first] = fixed[i].second;
  }
  // set unfixed in mapped index and set full index to start pos
  for (size_t i = 0, j = 0; i < m_fullIndex.size(); ++i)
  {
    if (m_fullIndex[i] == -1)
    {
      m_freeIndex.push_back(&m_fullIndex[i]);
      m_freeSizes[j] = m_fullIndex.sizes()[i];
      m_fullIndex[i] = 0;
      ++j;
    }
  }
}

const FixedIndex& FixedIndex::operator++()
{
  for (int64_t i = m_freeDim - 1; i != -1ll; --i)
  {
    ++*m_freeIndex[i];
    if (*m_freeIndex[i] != m_freeSizes[i])
      break;
    if (i == 0)
      m_atEnd = true;
    *m_freeIndex[i] = 0;
  }
  return *this;
}

//
FixedIndex::operator const Index&() const
{
  return m_fullIndex;
}

// allow read-only access to individual values
const int64_t& FixedIndex::operator[](size_t i) const
{
  return *m_freeIndex[i];
}

// allow modification of individual values
int64_t& FixedIndex::operator[](size_t i)
{
  return *m_freeIndex[i];
}


bool FixedIndex::end()
{
  return m_atEnd;
}

const std::vector<int64_t*>& FixedIndex::free() const
{
  return m_freeIndex;
}


const std::vector<int64_t>& FixedIndex::sizes() const
{
  return m_freeSizes;
}

