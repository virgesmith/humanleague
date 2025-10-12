// Index.h

#pragma once

#include <vector>

#include <cstddef>
#include <cstdint>

// Indexer for elements in n-D array - iterates over entire array
class Index
{
public:

  explicit Index(const std::vector<int64_t>& sizes);

  // Disable copy (purely because its not used, could re-enable if necessary)
  Index(const Index& rhs) = delete;

  virtual ~Index() {}

  // TODO return *this? (covariant return type)
  virtual const std::vector<int64_t>& operator++();

  // Implicitly cast to index vector
  operator const std::vector<int64_t>&() const;

  // TODO rename to dim
  size_t size() const;

  const std::vector<int64_t>& sizes() const;

  // allow read-only access to individual values
  const int64_t& operator[](size_t i) const;

  // allow modification of individual values
  int64_t& operator[](size_t i);

  void reset();

  bool end() const;

protected:
  size_t m_dim;
  std::vector<int64_t> m_idx;
  std::vector<int64_t> m_sizes;
  size_t m_storageSize;
  bool m_atEnd;
};


// Indexer for elements in n-D array, treating storage as column-major
class TransposedIndex : public Index
{
public:
  // Construct from array size
  explicit TransposedIndex(const std::vector<int64_t>& sizes);

  // overload increment operator
  const std::vector<int64_t>& operator++();

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
  size_t m_freeDim; // this is the dim of the free indices only
  Index m_fullIndex;
  std::vector<int64_t*> m_freeIndex;
  std::vector<int64_t> m_freeSizes;
  bool m_atEnd;

};
