#pragma once

#include "NDArray.h"

#include <vector>
#include <cassert>
#include <iostream>

template<typename T>
T sum(const std::vector<T>& v)
{
  return std::accumulate(v.begin(), v.end(), 0);
}

template<typename T>
T max(const std::vector<T>& v)
{
  return *std::max_element(v.begin(), v.end());
}

template<typename T>
void print(const std::vector<T>& v)
{
  for (size_t i = 0; i < v.size(); ++i)
  {
    std::cout << v[i] << " ";
  }
  std::cout << ";" << std::endl;
}

template<typename T>
void print(T* p, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    std::cout << p[i] << " ";
  }
  std::cout << ";" << std::endl;
}

template<size_t D, typename T>
T max(const NDArray<D, T>& v)
{
  return *std::max_element(v.rawData(), v.rawData() + v.storageSize());
}

// this assumes iterator is pointing to the start of the 1-d slice
template<size_t D, typename T, size_t O>
T min(typename NDArray<D,T>::template ConstIterator<O>& it) 
{
   T minVal = *it;
   while(!it.end())
   {
     minVal = std::min(minVal, *it);
     ++it;
   }
   return minVal;
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
uint32_t marginalProduct<1>(const std::vector<std::vector<uint32_t>>& m, const size_t* idx)
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

