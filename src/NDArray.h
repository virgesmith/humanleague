
#pragma once

#include <algorithm>
#include <cstddef>
#include <cassert>

//#include <Python.h>

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

    ConstIterator& operator++()
    {
      ++m_idx[Orient];
      return *this;
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

    Iterator& operator++()
    {
      ++m_idx[Orient];
      return *this;
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

  NDArray() : m_storageSize(0), m_data(0), m_owned(true)
  {
    m_sizes[0] = 0;
    m_sizesl[0] = 0;
  }

  NDArray(const size_t* sizes) : m_storageSize(0), m_data(0), m_owned(true)
  {
    resize(sizes);
  }
  
  // Construct with storage managed by some other object
  // TODO disable stuff like resize
  NDArray(long int* sizes, T* const storage)
  {
    std::copy(sizes, sizes + Dim, m_sizes);
    std::copy(sizes, sizes + Dim, m_sizesl);
    m_storageSize = sizes[0];
    assert(m_storageSize < MaxSize);
    for (size_t i = 1; i < Dim; ++i)
    {
      assert(sizes[i] < MaxSize);
      m_storageSize *= sizes[i];
    }

    m_data = storage;
    m_owned = false;
  }

  // Disallow copy
  NDArray(const NDArray&) = delete;

  ~NDArray()
  {
    if (m_owned)
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

  // python integration (TODO decide and remove one)
  const long* sizesl() const
  {
    return m_sizesl;
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
    if (m_owned)
    {
      size_t oldStorageSize = m_storageSize;

      std::copy(sizes, sizes + Dim, m_sizes);
      std::copy(sizes, sizes + Dim, m_sizesl);
      m_storageSize = sizes[0];
      assert(m_storageSize < MaxSize);
      for (size_t i = 1; i < Dim; ++i)
      {
        assert(sizes[i] < MaxSize);
        m_storageSize *= sizes[i];
      }

      // no realloc if storageSize unchanged
      if (m_storageSize > oldStorageSize)
      {
        deallocate(m_data);
        m_data = allocate(m_storageSize);
      }
    }
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

  value_type* begin() const
  {
    return m_data;
  }

  value_type* end() const
  {
    return m_data + m_storageSize;
  }

  // relinqish ownership
  void release()
  {
    m_data = nullptr;
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
    //return (T*)PyMem_Malloc(size * sizeof(T));
  }

  void deallocate(T* p) const
  {
    delete [] p;
    //PyMem_Free(p);
  }


private:

  size_t m_sizes[D];
  long m_sizesl[D];
  size_t m_storageSize;
  T* m_data;
  bool m_owned;
};

// 0d & 1d not implemented for obvious reasons
template<typename T> class NDArray<0, T>;
template<typename T> class NDArray<1, T>;


