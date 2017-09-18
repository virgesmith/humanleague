#pragma once

#include "NDArray2.h"
#include "Index2.h"

#include <vector>
#include <numeric>
#include <cassert>
#include <iostream>

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

// template<typename T, size_t O>
// std::vector<double> slice(const NDArray<2, T>& input, size_t index)
// {
//   if (index >= input.sizes()[O])
//     throw std::runtime_error("index out of bounds in slice");

//   // 1-O will give the non-orientation size for D=2
//   size_t remainingSize = input.sizes()[1 - O];

//   std::vector<T> output(remainingSize);
//   Index<2, O> inputIndex(input.sizes(), index);
//   for(size_t outputIndex = 0;!inputIndex.end(); ++inputIndex, ++outputIndex)
//   {
//     output[outputIndex] = input[inputIndex];
//   }
//   return output;
// }


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
