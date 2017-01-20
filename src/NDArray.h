
#pragma once

// Attempt 2 at n-D array

#include <algorithm>
#include <cstddef>
#include <cassert>

static const size_t Index_Unfixed = -1ull;


// Indexer for elements in n-D array, holding dimension C constant
// Use C=Index_Unfixed to loop over all elements
template<size_t D, size_t C>
class Index
{
public:

  static const size_t Dim = D;

  Index(const size_t* sizes) : m_atEnd(false) 
  {
    std::fill(m_idx, m_idx + Dim, 0);
    std::copy(sizes, sizes + D, m_sizes);
  }
  
  size_t* operator++()
  {
    for (size_t i = Dim - 1; i != -1ull; --i)
    {
      // ignore the iteration axis
      if (i == C) continue;
      
      ++m_idx[i];
      if (m_idx[i] != m_sizes[i])
        break;
      if (i == 0 || (C == 0 && i == 1)) 
        m_atEnd = true;
      m_idx[i] = 0;
    } 
    return m_idx;
  }
  
  // implicit cast
  operator size_t*()
  {
    return &m_idx[0];
  }
  
  bool end()
  {
    return m_atEnd;
  }
  

private:
  size_t m_idx[Dim];
  size_t m_sizes[Dim];
  bool m_atEnd;
};


// The array storage
template<size_t D, typename T>
class NDArray
{
public:

  static const size_t Dim = D;

  // Max size in any one dimension of ~1e9
  static const size_t MaxSize = 1u << 30;

  typedef T value_type;

  typedef const T& const_reference;

  typedef T& reference;

  // RW iterator over one dimension (O) of an n-D array given an index
  template<size_t O>
  class ConstIterator
  {
  public:

    static const size_t Orient = O;

    ConstIterator(const NDArray<D, T>& a, size_t* idx) : m_a(a)
    {
      // copy indices
      std::copy(idx, idx + D, m_idx);
      // set index of orientation dimension to zero
      m_idx[O] = 0;
    }

    ~ConstIterator() { }

    const size_t* idx() const
    {
      return m_idx;
    }

    void operator++()
    {
      ++m_idx[Orient];
    }

    bool end() const
    {
      return m_idx[Orient] >= m_a.size(Orient);
    }

    const_reference operator*() const
    {
      return m_a[m_idx];
    }

  private:
    const NDArray& m_a;
    size_t m_idx[D];
  };

  // RO iterator over one dimension (O) of an n-D array given an index
  template<size_t O>
  class Iterator
  {
  public:

    static const size_t Orient = O;

    Iterator(NDArray<D, T>& a, size_t* idx) : m_a(a)
    {
      // copy indices
      std::copy(idx, idx + D, m_idx);
      // set index of orientation dimension to zero
      m_idx[Orient] = 0;
    }

    ~Iterator() { }

    const size_t* idx() const
    {
      return m_idx;
    }

    void operator++()
    {
      ++m_idx[Orient];
    }

    bool end() const
    {
      return m_idx[Orient] >= m_a.size(Orient);
    }

    const_reference operator*() const
    {
      return m_a[m_idx];
    }

    reference operator*()
    {
      return m_a[m_idx];
    }

  private:
    NDArray& m_a;
    size_t m_idx[D];
  };

  NDArray() : m_storageSize(0), m_data(0)
  {
    m_sizes[0] = 0;
  }

  NDArray(const size_t* sizes)
  {
    resize(sizes);
  }
  
  // Disallow copy
  NDArray(const NDArray&) = delete;

  ~NDArray()
  {
    deallocate(m_data);
  }

  size_t size(size_t dim) const
  {
    assert(dim < Dim);
    return m_sizes[dim];
  }
  
  const size_t* sizes() const
  {
    return m_sizes;
  }

  size_t storageSize() const 
  {
    return m_storageSize;
  }
  
  const T* rawData() const
  {
    return m_data;
  }

  void resize(const size_t* sizes)
  {
    // TODO no realloc if storageSize unchanged
    deallocate(m_data);

    std::copy(sizes, sizes + Dim, m_sizes);
    m_storageSize = sizes[0];
    assert(m_storageSize < MaxSize);
    for (size_t i = 1; i < Dim; ++i)
    {
      assert(sizes[i] < MaxSize);
      m_storageSize *= sizes[i];
    }
    m_data = allocate(m_storageSize);
  }

  void assign(T val) const
  {
    std::fill(m_data, m_data + m_storageSize, val);
  }

  reference operator[](const size_t* index)
  {
    return m_data[offset(index)];
  }

  const_reference operator[](const size_t* index) const
  {
    return m_data[offset(index)];
  }

private:

  size_t offset(const size_t* const idx) const
  {
    // TODO this is pretty horrible, but it works. Template it?
    size_t ret = 0;
    size_t mult = m_storageSize;
    for (size_t i = 0; i < Dim; ++i)
    {
      mult /= m_sizes[i];
      ret += mult * idx[i];
    }
    return ret;
  }

  T* allocate(size_t size) const
  {
    return new T[size];
  }

  void deallocate(T* p) const
  {
    delete [] p;
  }


private:

  size_t m_sizes[D];
  size_t m_storageSize;
  T* m_data;

};

// 0d not implemented for obvious reasons
template<typename T> class NDArray<0, T>;


