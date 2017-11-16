// Index.h

#pragma once

#include <vector>

#include <cstddef>
#include <cstdint>

// Indexer for elements in n-D array, optionally holding one dimension constant
class Index
{
public:
  static const int64_t Unfixed = -1;

  // Omit the second argument to loop over all elements
  // TODO remove the fixed index entirely from this class - use FixedIndex instead
  explicit Index(const std::vector<int64_t>& sizes, const std::pair<int64_t, int64_t>& fixed = {-1, -1});
  
  // Create an index with a predefined position (all dims unfixed)
  Index(const std::vector<int64_t>& sizes, const std::vector<int64_t>& values);
  
  Index(const Index& rhs);

  // TODO return *this
  const std::vector<int64_t>& operator++();

  // Implicitly cast to index vector
  operator const std::vector<int64_t>&() const;

  // TODO rename to dim
  size_t size() const;

  const std::vector<int64_t>& sizes() const;

  // allow read-only access to individual values
  const int64_t& operator[](size_t i) const;

  // allow modification of individual values
  int64_t& operator[](size_t i);

  // need this for e.g. R where storage is column-major
  // NB row-major offset calc is in NDArray itself
  size_t colMajorOffset() const;

  void reset();

  bool end() const;

protected:
  size_t m_dim;
  std::vector<int64_t> m_idx;
  std::vector<int64_t> m_sizes;
  // Fixed point (dim, idx)
  std::pair<int64_t, int64_t> m_fixed;
  size_t m_storageSize;
  bool m_atEnd;
};


// Indexer for elements in n-D array, treating storage as column-major
class TransposedIndex : public Index
{
public:
  // Omit the second argument to loop over all elements
  explicit TransposedIndex(const std::vector<int64_t>& sizes) : Index(sizes/*std::vector<int64_t>(sizes.rbegin(), sizes.rend())*/) { }

  const std::vector<int64_t>& operator++()
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
  
};



// Contains a mapping from a higher dimensionality to a lower one
class MappedIndex
{
public:
  MappedIndex(const Index& idx, const std::vector<int64_t>& mappedDimensions);
  
  const MappedIndex& operator++();  

  // TODO better to overload NDArray to take Index types???
  operator const std::vector<int64_t*>&() const;

  // allow read-only access to individual values
  const int64_t& operator[](size_t i) const;
  
  // allow modification of individual values
  int64_t& operator[](size_t i);
  
  bool end();

  //static std::vector<int64_t> excludeFrom(const std::vector<int64_t>& dims, int64_t excludedDim);

private:
  size_t m_dim;
  std::vector<int64_t> m_sizes;
  std::vector<int64_t*> m_mappedIndex;
  bool m_atEnd;
};

class FixedIndex
{
public:
  // Loop over elements with some dimensions fixed
  FixedIndex(const std::vector<int64_t>& sizes, const std::vector<std::pair<int64_t, int64_t>>& fixed);

  // increment
  const FixedIndex& operator++();

  // Returns the FULL index
  operator const Index&() const;

  // allow read-only access to individual unfixed values
  const int64_t& operator[](size_t i) const;
  
  // allow modification of individual unfixed values
  int64_t& operator[](size_t i);
  
  const std::vector<int64_t>& sizes() const;

  bool end();

  const std::vector<int64_t*>& free() const;

private:
  size_t m_dim; // this is the dim of the free indices only
  Index m_fullIndex;
  std::vector<int64_t*> m_freeIndex;
  std::vector<int64_t> m_sizes;
  bool m_atEnd;

};