#pragma once

#include "NDArrayOld.h"
#include "IndexOld.h"

#include <vector>
#include <numeric>
#include <cassert>
#include <iostream>

namespace old {

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

template<size_t D>
uint32_t marginalProduct(const std::vector<std::vector<uint32_t>>& m, const size_t* idx)
{
  return m[D-1][idx[D-1]] * marginalProduct<D-1>(m, idx);
}

template<>
inline uint32_t marginalProduct<1>(const std::vector<std::vector<uint32_t>>& m, const size_t* idx)
{
  return m[0][idx[0]];
}

template<size_t D, typename T, size_t O>
std::vector<T> reduce(const NDArray<D, T>& input)
{
  // check valid orientation
  assert(O < (NDArray<D, T>::Dim));

  std::vector<T> sums(input.size(O), 0);

  //size_t idx[NDArray<D, T>::Dim] = { 0 };
  Index<D,O> indexer(input.sizes());

  for (; !indexer.end(); ++indexer)
  {
    typename NDArray<D, T>:: template ConstIterator<O> it(input, indexer.operator size_t*());
    for(size_t i = 0; !it.end(); ++it, ++i)
    {
      sums[i] += *it;
    }
  }

  return sums;
}

// take a D-1 dimensional slice at element index in orientation O
template<size_t D, typename T, size_t O>
NDArray<D-1, T> slice(const NDArray<D, T>& input, size_t index)
{
  if (index >= input.sizes()[O])
    throw std::runtime_error("index out of bounds in slice");

  size_t remainingSizes[D-1];
  for (size_t i = 0, j = 0; i < D; ++i)
  {
    if (i != O) 
    {
      remainingSizes[j] = input.sizes()[i];
      ++j;
    }
  }
  NDArray<D-1, T> output(remainingSizes);
  Index<D, O> inputIndex(input.sizes(), index);
  Index<D-1, Index_Unfixed> outputIndex(output.sizes());
  for(;!inputIndex.end(); ++inputIndex, ++outputIndex)
  {
    output[outputIndex] = input[inputIndex];
  }
  return output;
}

template<typename T, size_t O>
std::vector<double> slice(const NDArray<2, T>& input, size_t index)
{
  if (index >= input.sizes()[O])
    throw std::runtime_error("index out of bounds in slice");

  // 1-O will give the non-orientation size for D=2
  size_t remainingSize = input.sizes()[1 - O];

  std::vector<T> output(remainingSize);
  Index<2, O> inputIndex(input.sizes(), index);
  for(size_t outputIndex = 0;!inputIndex.end(); ++inputIndex, ++outputIndex)
  {
    output[outputIndex] = input[inputIndex];
  }
  return output;
}


// Converts a D-dimensional population array into a list with D columns and pop rows
template<size_t D>
std::vector<std::vector<int>> listify(const size_t pop, const NDArray<D, uint32_t>& t)
{
  std::vector<std::vector<int>> list(D, std::vector<int>(pop));
  Index<D, Index_Unfixed> index(t.sizes());

  size_t pindex = 0;
  while (!index.end() /*&& pindex < 10*/) // TODO fix inf loop!
  {
    for (size_t i = 0; i < t[index]; ++i)
    {
      for (size_t j = 0; j < D; ++j)
      {
        list[j][pindex] = index[j];
      }
      ++pindex;
    }
    ++index;
  }
  return list;
}

// picks a 1d slice which won't go -ve if you subtract the residual
template<size_t D, size_t O>
Index<D, O> pickIndex(const std::vector<int32_t>& r, const NDArray<D, uint32_t>& t, bool& willGoNegative)
{
  //std::cout << "pickIndex: O = " << O << " r.size=" << r.size() << std::endl;
  Index<D, O> idx(t.sizes());
  Index<D, O> leastNegativeIndex(t.sizes());

  int32_t leastNegativeVal = std::numeric_limits<int32_t>::min();
  while (!idx.end())
  {
    //print(idx.m_idx, D);
    typename NDArray<D, uint32_t>::template ConstIterator<O> it(t, idx);
    int32_t minVal = static_cast<int32_t>(*it) - r[0];
    //std::cout << static_cast<int32_t>(*it) - r[0];
    ++it;
    for (size_t i = 1; i < r.size(); ++i)
    {
      //std::cout << "," << static_cast<int32_t>(*it) - r[i];
      minVal = std::min(minVal, static_cast<int32_t>(*it) - r[i]);
      ++it;
    }
    //std::cout << std::endl;

    //std::cout << counter << " : min_t" << " = " << minVal << std::endl;

    if (minVal >= 0)
    {
      //std::cout << "DIR:" << O << " can adjust without going -ve" << std::endl;
      //print(idx.m_idx, D);
      willGoNegative = false;
      return idx;
    }
    else if (minVal > leastNegativeVal)
    {
      leastNegativeVal = minVal;
      leastNegativeIndex = idx;
      //std::cout << "least -ve = " << minVal << " at index ";
      //print(idx.m_idx, D);
    }
    ++idx;
  }
  //std::cout << "DIR:" << O  << " CANT adjust without going -ve: " << leastNegativeVal << std::endl;
  //print(leastNegativeIndex.m_idx, D);
  willGoNegative = true;
  return leastNegativeIndex;
}


template<size_t D, size_t O>
bool adjust(const std::vector<std::vector<int32_t>>& rs, NDArray<D, uint32_t>& t, bool allowNegative)
{
  //std::cout << "adjust" << O << " -ve=" << allowNegative << std::endl;
  const std::vector<int32_t>& r = rs[O];
  // pick any index s.t. subtracting r won't result in -ve values,
  // or otherwise the index that will result in the least negative value
  bool willGoNegative;
  Index<D, O> idx = pickIndex<D, O>(r, t, willGoNegative);

  if (!allowNegative && willGoNegative)
    return false;

  typename NDArray<D, uint32_t>::template Iterator<O> it(t, idx);

  bool floored = false;
  for(size_t i = 0; !it.end(); ++it, ++i)
  {
  //   //std::cout << i << " ";
  //   // floor at zero
    int32_t newVal = static_cast<int32_t>(*it) - r[i];
  //   // this is causing large values in adj resid (somehow)
  //   // if (newVal < 0)
  //   // {
  //   //   floored = true;
  //   //   newVal = 0;
  //   // }
  //   //std::cout << *it << "->" << newVal << std::endl;
    *it = newVal;
  }
  //std::cout << std::endl;
  return !floored;
}

}
