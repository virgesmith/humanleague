
#pragma once

#include <vector>
#include <random>
#include <stdexcept>
#include <initializer_list>
#include <algorithm>

#include <iostream>

#include <cstdlib>


#define CHECK_BOUNDS(i) \
 if (i >= m_data.size()) \
    throw std::runtime_error(std::string("bounds " __FILE__ ":") + std::to_string(__LINE__))

// unsigned -ve test (i.e. check MSB=1)
inline bool uneg(uint32_t i)
{
  return i & (1<<31);
}



//template<size_t D>
//class Iterator
//{
//public:

//  static const size_t Dim = D;

//  // In 1D case, nothing to iterate over (?)
//  Iterator(size_t dim, size_t val, size_t* sizes) : m_idxDim(dim), m_sizes(sizes, sizes + Dim), m_idx(Dim, 0), m_end(Dim == 1)
//  {
//    m_idx[m_idxDim] = val;
//  }

//  Iterator& operator++()
//  {
//    for (size_t i = 0; i < Dim; ++i)
//    {
//      if (i != m_idxDim)
//      {
//        ++m_idx[i];
//        if (m_idx[i] < m_sizes[i])
//        {
//          return *this;
//        }
//        else
//        {
//          // if not last dimension, reset
//          if (i != Dim - 1)
//            m_idx[i] = 0;
//          // otherwise we are at end
//          else
//            m_end = true;
//        }
//      }
//    }
//    return *this;
//  }

//  bool end() const
//  {
//    return m_end;
////    for (size_t i = 0; i < Dim; ++i)
////    {
////      if (i != m_idxDim)
////      {
////        if (m_idx[i] < m_sizes[i] - 1)
////          return false;
////      }
////    }
////    return true;
//  }

//  const std::vector<size_t> val() const
//  {
//    return m_idx;
//  }

//private:

//  size_t m_idxDim;

//  std::vector<size_t> m_sizes;
//  std::vector<size_t> m_idx;
//  bool m_end;

//};


template<size_t D, typename T>
class Tensor
{
public:

  // the dimension
  static const size_t Dim = D;

  // the ultimate stored type
  typedef T innermost_value_type;

  // the next dimension down
  typedef Tensor<D-1, T> value_type;

  // this dimension
  typedef std::vector<value_type> container_type;

  typedef typename container_type::iterator iterator;

  typedef typename container_type::const_iterator const_iterator;

  // TODO see if can get rid of this?
  Tensor() { }

  Tensor(size_t* sizes) : m_data(sizes[0], value_type(sizes+1))
  {
    // assumes sizes.size() == D
  }

  Tensor(const Tensor<D,T>& rhs)
  {
    m_data = rhs.m_data; // check this works ok
  }

  size_t dim() const { return Dim; }

  void sizes(std::vector<size_t>& s) const
  {
    s.push_back(m_data.size());
    m_data[0].sizes(s);
  }

  size_t size() const { return m_data.size(); }

  // resize without altering content
  void resize(size_t* sizes)
  {
    m_data.resize(sizes[0]);
    for (size_t i = 0; i < m_data.size(); ++i)
    {
      m_data[i].resize(sizes+1);
    }
  }

  // alter content without resizing
  void assign(const T& v)
  {
    for (size_t i = 0; i < m_data.size(); ++i)
    {
      m_data[i].assign(v);
    }
  }

  iterator begin() { return m_data.begin(); }

  iterator end() { return m_data.end(); }

  const_iterator begin() const { return m_data.begin(); }

  const_iterator end() const { return m_data.end(); }

  value_type& operator[](size_t i) { CHECK_BOUNDS(i); return m_data[i]; }

  const value_type& operator[](size_t i) const { CHECK_BOUNDS(i); return m_data[i]; }

  innermost_value_type& at(uint32_t* idx) { return m_data[idx[0]].at(idx+1); }

  const innermost_value_type& at(uint32_t* idx) const { return m_data[idx[0]].at(idx+1); }

  uint32_t sum() const
  {
    uint32_t s = 0u;
    sumImpl(s);
    return s;
  }

private:

  // allows next higher dim to access sumImpl
  friend class Tensor<D+1, T>;

  void sumImpl(uint32_t& s) const
  {
    for (size_t i = 0; i < m_data.size(); ++i)
    {
      m_data[i].sumImpl(s);
    }
  }

  container_type m_data;
};

template<size_t D, typename T>
bool operator==(const Tensor<D, T>& x, const Tensor<D, T>& y)
{
  if (x.size() != y.size())
    return false;

  for (size_t i = 0; i < x.size(); ++i)
  {
    if (!(x[i] == y[i]))
      return false;
  }
  return true;
}

template<size_t D, typename T>
bool operator!=(const Tensor<D, T>& x, const Tensor<D, T>& y)
{
  return !(x == y);
}


//// TODO generalise to N-D
//template<typename T>
//Tensor<1, std::make_signed<T>> operator-(const Tensor<1, T>& x, const Tensor<1, T>& y)
//{
//  size_t size = x.size();
//  Tensor<1,std::make_signed<T>> result(&size);
//  // TODO check x.size() -== y.size()
//  for (size_t i = 0; i < x.size(); ++i)
//  {
//    result[i] = x[i] - y[i];
//  }
//  return result;
//}


// TODO generalise to N-D
template<typename T>
bool isZero(const Tensor<1, T>& x)
{
  size_t size = x.size();

  for (size_t i = 0; i < size; ++i)
  {
    if (x[i] != 0)
      return false;
  }
  return true;
}

template<typename T>
bool isZero(const Tensor<2, T>& x)
{

  for (size_t i = 0; i < x.size(); ++i)
  {
    for (size_t j = 0; j < x[0].size(); ++j)
      if (x[i][j] != 0)
        return false;
  }
  return true;
}

template<size_t D, typename T>
T min(const Tensor<D, T>& x)
{
  T val = std::numeric_limits<T>::max();
  
  for (size_t i = 0; i < x.size(); ++i)
  {
    val = std::min(val, min(x[i]));
  }
  return val;
}

template<size_t D, typename T>
T max(const Tensor<D, T>& x)
{
  T val = std::numeric_limits<T>::min();
  
  for (size_t i = 0; i < x.size(); ++i)
  {
    val = std::max(val, max(x[i]));
  }
  return val;
}

template<typename T>
T min(const Tensor<2, T>& x, size_t idx)
{
  T val = std::numeric_limits<T>::max();
  
  for (size_t i = 0; i < x.size(); ++i)
  {
    val = std::min(val, x[i][idx]);
  }
  return val;
}

template<typename T>
T min(const Tensor<1, T>& x)
{
  return *std::min_element(x.begin(), x.end());
}

template<typename T>
T max(const Tensor<1, T>& x)
{
  return *std::max_element(x.begin(), x.end());
}

template<typename T>
class Tensor<0, T>; // undefined for zero dimensions

// 1D partial specialisation
template<typename T>
class Tensor<1, T>
{
public:

  typedef T value_type;

  typedef std::vector<value_type> container_type;

  typedef typename container_type::iterator iterator;

  typedef typename container_type::const_iterator const_iterator;

  Tensor() { }

  // zero-initialises
  Tensor(size_t* size) : m_data(*size, 0)
  {
  }

  // zero-initialises
  Tensor(size_t size) : m_data(size, 0)
  {
  }

  // zero-initialises
  Tensor(std::initializer_list<T> data) : m_data(data)
  {
  }

  template<typename I>
  Tensor(I b, I e) : m_data(b, e)
  {
  }

  Tensor(const Tensor<1, T>& rhs) : m_data(rhs.m_data)
  {
  }

  size_t dim() const { return 1; }

  size_t size() const { return m_data.size(); }

  void sizes(std::vector<size_t>& s) const
  {
    s.push_back(m_data.size());
  }

  void resize(size_t* sizes)
  {
    m_data.resize(sizes[0]);
  }

  void assign(const T& v)
  {
    m_data.assign(m_data.size(), v);
  }

  iterator begin() { return m_data.begin(); }

  iterator end() { return m_data.end(); }

  const_iterator begin() const { return m_data.begin(); }

  const_iterator end() const { return m_data.end(); }

  value_type& operator[](size_t i) { CHECK_BOUNDS(i); return m_data[i]; }

  const value_type& operator[](size_t i) const { CHECK_BOUNDS(i); return m_data[i]; }

  value_type& at(const uint32_t* idx) { return m_data[idx[0]]; }

  const value_type& at(const uint32_t* idx) const { return m_data[idx[0]]; }

  uint32_t sum() const
  {
    uint32_t s = 0u;
    sumImpl(s);
    return s;
  }

private:

  // allows next higher dim to access sumImpl
  friend class Tensor<2, T>;

  void sumImpl(uint32_t& s) const
  {
    s = std::accumulate(m_data.begin(), m_data.end(), s);
  }

  container_type m_data;
};


//template<size_t D, typename T> class Tensor;

template<size_t D, typename T>
Tensor<D-1, T> reduce(size_t dim, const Tensor<D, T>& input);


template<typename T>
Tensor<1, T> reduce(size_t dim, const Tensor<2, T>& input)
{
  // TODO check dim < D

  if (dim == 0)
  {
    size_t size = input.size();
    Tensor<1, uint32_t> sums(&size);
    for (size_t i = 0; i < input.size(); ++i)
    {
      sums[i] = std::accumulate(input[i].begin(), input[i].end(), 0u);
    }
    return sums;
  }
  else //if (dim == 1)
  {
    size_t size = input[0].size();
    Tensor<1, uint32_t> sums(&size);
    for (size_t i = 0; i < input[0].size(); ++i)
    {
      for (size_t j = 0; j < input.size(); ++j)
      {
        sums[i] += input[j][i];
      }
    }
    return sums;
  }
}

template<typename T>
Tensor<1, T> reduce(size_t dim, const Tensor<3, T>& input)
{
  // TODO check dim < D
  Tensor<1, uint32_t> sums;
  if (dim == 0)
  {
    size_t size = input.size();
    //std::cout << sizes[0] << ", " << sizes[1] << std::endl;
    sums.resize(&size);
    sums.assign(0);
    for (size_t i = 0; i < input[0][0].size(); ++i)
    {
      for (size_t j = 0; j < input[0].size(); ++j)
      {
        for (size_t k = 0; k < input.size(); ++k)
          sums[k] += input[k][j][i];
      }
    }
    return sums;
  }
  else if (dim == 1)
  {
    size_t size = input[0].size();
    //std::cout << sizes[0] << ", " << sizes[1] << std::endl;
    sums.resize(&size);
    sums.assign(0);
    for (size_t i = 0; i < input.size(); ++i)
    {
      for (size_t j = 0; j < input[0][0].size(); ++j)
      {
        for (size_t k = 0; k < input[0].size(); ++k)
          sums[k] += input[i][k][j];
      }
    }
    return sums;
  }
  else //if (dim == 2)
  {
    size_t size = input[0][0].size();
    //std::cout << sizes[0] << ", " << sizes[1] << std::endl;
    sums.resize(&size);
    sums.assign(0);
    for (size_t i = 0; i < input[0].size(); ++i)
    {
      for (size_t j = 0; j < input.size(); ++j)
      {
        for (size_t k = 0; k < input[0][0].size(); ++k)
          sums[k] += input[j][i][k];
      }
    }
    return sums;
  }
  return sums;
}



