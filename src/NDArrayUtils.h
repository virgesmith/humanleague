#pragma once

#include "NDArray.h"
#include "Index.h"

#include <vector>
#include <numeric>
#include <cassert>
#include <iostream>
#include <cmath>


int32_t maxAbsElement(const std::vector<int32_t>& r);

std::vector<int32_t> diff(const std::vector<uint32_t>& x, const std::vector<uint32_t>& y);

template<typename T, typename U>
void diff(const NDArray<T>& x, const NDArray<U>& y, NDArray<double>& d)
{
  // TODO check x y and d sizes match
  for (Index index(x.sizes()); !index.end(); ++index)
  {
    d[index] = x[index] - y[index];
  }
}

bool allZeros(const std::vector<std::vector<int32_t>>& r);

template<typename T>
T sum(const std::vector<T>& v)
{
  return std::accumulate(v.begin(), v.end(), T(0));
}

template<typename T>
T product(const std::vector<T>& v)
{
  return std::accumulate(v.begin(), v.end(), T(1), std::multiplies<T>());
}

template<typename T>
T sum(const NDArray<T>& a)
{
  return std::accumulate(a.rawData(), a.rawData() + a.storageSize(), T(0));
}

template<typename T>
T min(const NDArray<T>& a)
{
  T minVal = std::numeric_limits<T>::max();
  for (Index i(a.sizes()); !i.end(); ++i)
  {
    minVal = std::min(minVal, a[i]);
  }
  return minVal;
}

template<typename T>
T max(const NDArray<T>& a)
{
  T minVal = std::numeric_limits<T>::max();
  for (Index i(a.sizes()); !i.end(); ++i)
  {
    minVal = std::min(minVal, a[i]);
  }
  return minVal;
}

// TODO move printing somehwere else
template<typename T>
void print(const std::vector<T>& v, std::ostream& ostr = std::cout)
{
  for (size_t i = 0; i < v.size(); ++i)
  {
    ostr << v[i] << ", ";
  }
  ostr << std::endl;
}

template<typename T>
void print(T* p, size_t n, size_t breakAt = 1000000, std::ostream& ostr = std::cout)
{
  for (size_t i = 0; i < n; ++i)
  {
    ostr << p[i] << ", ";
    if (!((i+1) % breakAt))
      ostr << std::endl;
  }
  ostr << std::endl;
}

template<typename T>
bool isZero(const std::vector<T>& v)
{
  for (size_t i = 0; i < v.size(); ++i)
  {
    if (v[i] != 0)
      return false;
  }
  return true;
}

template<typename T>
T marginalProduct(const std::vector<std::vector<T>>& m, const std::vector<int64_t>& idx)
{
  assert(m.size() == idx.size());
  T p = T(1);

  for (size_t d = 0; d < m.size(); ++d)
  {
    p *= m[d][idx[d]];
  }
  return p;
}

// Reduce n-D array to 1-D sums.
// NB if the input is already 1d this just returns the original NDArray as a vector
template<typename T>
std::vector<T> reduce(const NDArray<T>& input, size_t orient)
{
  // return reduced;
  // check valid orientation
  if (!(orient < input.dim()))
    throw std::runtime_error("reduce dimension " + std::to_string(orient)
                                                 + " greater than array dimension "
                                                 + std::to_string(input.dim()));

  std::vector<T> sums(input.size(orient), 0);

  // this is a bottleneck slower according to callgrind, but VTune says otherwise
  Index indexer(input.sizes());
  for (; !indexer.end(); ++indexer)
  {
    sums[indexer[orient]] += input[indexer];
  }

  return sums;
}


// Reduce n-D array to m-D sums (where m<n)
template<typename T>
NDArray<T> reduce(const NDArray<T>& input, const std::vector<int64_t>& preservedDims)
{
  const size_t reducedDim = preservedDims.size();
  // check valid orientation
  assert(reducedDim < input.dim());

  std::vector<int64_t> preservedSizes(reducedDim);
  for (size_t d = 0; d < reducedDim; ++d)
  {
    preservedSizes[d] = input.sizes()[preservedDims[d]];
  }

  NDArray<T> reduced(preservedSizes);
  reduced.assign(T(0));

  Index index(input.sizes());
  MappedIndex rIndex(index, preservedDims);
  for (; !index.end(); ++index)
  {
    reduced[rIndex] += input[index];
  }

  return reduced;
}


// take a D-1 dimensional slice at element index in orientation O
template<typename T>
NDArray<T> slice(const NDArray<T>& input, std::pair<int64_t, int64_t> index)
{
  return slice(input, std::vector<std::pair<int64_t, int64_t>>(1, index));
}

template<typename T>
NDArray<T> slice(const NDArray<T>& outer, const std::vector<std::pair<int64_t, int64_t>>& fixedDims)
{
  for (size_t i = 0; i < fixedDims.size(); ++i)
  {
    if ((size_t)fixedDims[i].first >= outer.dim())
      throw std::runtime_error("dimension out of bounds in slice");
    if (fixedDims[i].second >= outer.sizes()[fixedDims[i].first])
      throw std::runtime_error("index out of bounds in slice");
  }
  // if dims empty have to make a complete copy of marginal and return it, which is massively inefficient
  if (fixedDims.empty())
  {
    NDArray<T> copy(outer.sizes());
    std::copy(outer.rawData(), outer.rawData() + outer.storageSize(), const_cast<T*>(copy.rawData()));
    return copy;
  }

  FixedIndex fixedIndex(outer.sizes(), fixedDims);

  // fixedIndex.sizes() returns the sizes of the free dimensions
  NDArray<T> sliced(fixedIndex.sizes());
  for(; !fixedIndex.end(); ++fixedIndex)
  {
    sliced[fixedIndex.free()] = outer[fixedIndex.operator const Index &()];
  }
  return sliced;
}

// Converts a D-dimensional population array into a list with D columns and pop rows
template<typename T>
std::vector<std::vector<int>> listify(const size_t pop, const NDArray<T>& t, int offset = 0)
{
  // use offset = 1 for langauges where array indexing begins at 1 (e.g. R)
  std::vector<std::vector<int>> list(t.dim(), std::vector<int>(pop));

  Index index(t.sizes());

  size_t pindex = 0;
  while (!index.end())
  {
    for (T i = 0; i < t[index]; ++i)
    {
      const std::vector<int64_t>& ref = index;
      for (size_t j = 0; j < t.dim(); ++j)
      {
        list[j][pindex] = offset + static_cast<int>(ref[/*t.dim() - 1 - */j]);
      }
      ++pindex;
    }
    ++index;
  }
  return list;
}

