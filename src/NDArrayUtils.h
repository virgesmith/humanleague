#pragma once

#include "NDArray.h"
#include "Index.h"

#include <vector>
#include <cassert>
#include <iostream>

template<typename T>
T sum(const std::vector<T>& v)
{
  return std::accumulate(v.begin(), v.end(), 0);
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


// picks a 1d slice which won't go -ve if you subtract the residual
template<size_t D, size_t O>
Index<D, O> pickIndex(const std::vector<int32_t>& r, const NDArray<D, uint32_t>& t)
{
  Index<D, O> idx(t.sizes());

  while (!idx.end())
  {
    typename NDArray<D, uint32_t>::template ConstIterator<O> it(t, idx);
    int32_t minVal = static_cast<int32_t>(*it) - r[0];
    ++it;
    for (size_t i = 1; i < r.size(); ++i)
    {
      minVal = std::min(minVal, static_cast<int32_t>(*it) - r[i]);
      ++it;
    }

    //std::cout << "min_t" << " = " << minVal << std::endl;
    
    if (minVal >= 0)
      break;
    ++idx;
  }
  // if no index found idx.end() == true
  return idx;
}


template<size_t D, size_t O>
bool adjust(const std::vector<int32_t>& r, NDArray<D, uint32_t>& t)
{
  // pick an index s.t. subtracting r won't result in -ve values
  Index<D, O> idx = pickIndex<D, O>(r, t);

  if (idx.end())
    return false;

  typename NDArray<D, uint32_t>::template Iterator<O> it(t, idx);

  for(size_t i = 0; !it.end(); ++it, ++i)
  {
    *it -= r[i];
  }
  return true;
}

std::vector<int32_t> diff(const std::vector<uint32_t>& x, const std::vector<uint32_t>& y)
{
  size_t size = x.size();
  assert(size == y.size());

  std::vector<int32_t> result(size);

  for (size_t i = 0; i < x.size(); ++i)
  {
    result[i] = x[i] - y[i];
  }
  return result;
}



