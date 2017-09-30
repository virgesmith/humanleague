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
  MappedIndex(const Index& idx, const std::vector<int64_t>& mappedDimensions);

  const MappedIndex& operator++();  

  // TODO better to overload NDArray to take Index types???
  operator const std::vector<int64_t*>&() const;

  bool end();
  
private:
  size_t m_dim;
  std::vector<int64_t> m_sizes;
  std::vector<int64_t*> m_mappedIndex;
  bool m_atEnd;
};

