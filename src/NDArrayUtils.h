#pragma once

#include "NDArray.h"
#include "Index.h"

#include <vector>
#include <numeric>
#include <cassert>
#include <iostream>


int32_t maxAbsElement(const std::vector<int32_t>& r);

std::vector<int32_t> diff(const std::vector<uint32_t>& x, const std::vector<uint32_t>& y);
std::vector<double> diff(const std::vector<double>& x, const std::vector<double>& y);

bool allZeros(const std::vector<std::vector<int32_t>>& r);

template<typename T>
T sum(const std::vector<T>& v)
{
  return std::accumulate(v.begin(), v.end(), 0);
}

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

namespace wip {

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

// Reduce n-D array to 1-D sums
template<typename T>
std::vector<T> reduce(const wip::NDArray<T>& input, size_t orient)
{
  // check valid orientation
  assert(orient < input.dim());

  std::vector<T> sums(input.size(orient), 0);

  Index indexer(input.sizes(), { orient, 0 });

  for (; !indexer.end(); ++indexer)
  {
    // Pass index in directly to avoid
    typename NDArray<T>::ConstIterator it(input, orient, indexer);
    for(size_t i = 0; !it.end(); ++it, ++i)
    {
      sums[i] += *it;
    }
  }

  return sums;
}

// Reduce n-D array to m-D sums (where m<n)
template<typename T>
wip::NDArray<T> reduce(const wip::NDArray<T>& input, const std::vector<int64_t>& preservedDims)
{
  const size_t reducedDim = preservedDims.size();
  // check valid orientation
  assert(reducedDim < input.dim());

  std::vector<int64_t> preservedSizes(reducedDim);
  for (size_t d = 0; d < reducedDim; ++d)
  {
    preservedSizes[d] = input.sizes()[preservedDims[d]];
  }

  wip::NDArray<T> reduced(preservedSizes);
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
wip::NDArray<T> slice(const wip::NDArray<T>& input, std::pair<int64_t, int64_t> index)
{
  if ((size_t)index.first >= input.dim())
    throw std::runtime_error("dimension out of bounds in slice");
  if (index.second >= input.sizes()[index.first])
    throw std::runtime_error("index out of bounds in slice");

  std::vector<int64_t> remainingSizes;
  remainingSizes.reserve(input.dim() - 1);
  for (size_t i = 0; i < input.dim(); ++i)
  {
    if (i != (size_t)index.first)
    {
      remainingSizes.push_back(input.sizes()[i]);
    }
  }
  NDArray<T> output(remainingSizes);
  Index inputIndex(input.sizes(), index);
  Index outputIndex(output.sizes());
  for(;!inputIndex.end(); ++inputIndex, ++outputIndex)
  {
    output[outputIndex] = input[inputIndex];
  }
  return output;
}


// Converts a D-dimensional population array into a list with D columns and pop rows
inline std::vector<std::vector<int>> listify(const size_t pop, const wip::NDArray<uint32_t>& t)
{
  std::vector<std::vector<int>> list(t.dim(), std::vector<int>(pop));
  wip::Index index(t.sizes());

  size_t pindex = 0;
  while (!index.end())
  {
    for (size_t i = 0; i < t[index]; ++i)
    {
      const std::vector<int64_t>& ref = index;
      for (size_t j = 0; j < t.dim(); ++j)
      {
        list[j][pindex] = ref[j];
      }
      ++pindex;
    }
    ++index;
  }
  return list;
}


}
