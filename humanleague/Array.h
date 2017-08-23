
#pragma once 

#include "Object.h"
#include "src/NDArray.h"

#include <Python.h>
#include <numpy/arrayobject.h>

#include <cstring>


// TODO use boost.python (requires >= 1.63, 16.4 comes with 1.58)
// or https://github.com/ndarray/Boost.NumPy
//#include <boost/python/numpy.hpp>

// See https://docs.scipy.org/doc/numpy/reference/c-api.html

namespace pycpp {

  // Utilities for numpy API
  inline void numpy_init() 
  {
    // import_array is an evil macro that for python3+ expands to a code block with a 
    // single if statement containing a (conditional) return statement, so not all paths return a value. 
    // The return value is essentially useless since it is only defined for success, thus no way of detecting errors. 
    // To workaround we wrap in a lambda, adding a non-conditional return statement and then ignoring the value. 
    []() -> void* { 
      import_array();
      return nullptr;
    }();
  }
    
  template<typename T> struct NpyType;

  template<> struct NpyType<double> { static const int Type = NPY_DOUBLE; };
  template<> struct NpyType<int> { static const int Type = NPY_INT; };
  template<> struct NpyType<uint32_t> { static const int Type = NPY_UINT32; };
  template<> struct NpyType<int64_t> { static const int Type = NPY_INT64; };
  template<> struct NpyType<bool> { static const int Type = NPY_BOOL; };

  // numpy arrays 
  template<typename T>
  class Array : public Object
  {
  public:
    typedef T value_type;

    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
  
    // uninitialised array with given dimension and sizes
    explicit Array(size_t dim, npy_intp* sizes) 
      : Object(PyArray_SimpleNew(dim, sizes, NpyType<T>::Type)) 
    { 
      PyArray_FILLWBYTE((PyArrayObject*)m_obj, 0);
    }

    // construct from an incoming numpy object
    explicit Array(PyObject* array) : Object(array)
    {
    }
    
    // Construct 1D from vector<T> 
    explicit Array(const std::vector<T>& a) : Array(a, a.size())
    {
      // see below constructor
    }

private: // this is a hack to chain constructors to avoid having to explicitly pass an addressable size value 
    // Construct 1D from vector<T> 
    explicit Array(const std::vector<T>& a, npy_intp size) : Array(1, &size)
    {
      std::memcpy(PyArray_GETPTR1((PyArrayObject*)m_obj, 0), a.data(), size * sizeof(T));
    }
    
public:
    // Construct from NDArray<D,T>. Data is presumed to be copied
    template<size_t D>
    explicit Array(NDArray<D, T>&& a) 
      : Object(PyArray_SimpleNewFromData(D, 
                                         const_cast<npy_intp*>(a.sizesl()), 
                                         NpyType<T>::Type, 
                                         const_cast<void*>((const void*)a.rawData()))) 
    {
      // memory ownership transferred?
      // leak if you dont, memory corruption if you do
      a.release();
    }
    
    ~Array()
    {
      // do we need to delete/decref?
    }
    
    // generic n-D access??
    const_reference& operator[](npy_intp* index) const
    {
      return *(const_pointer)PyArray_GetPtr(const_cast<PyArrayObject*>(reinterpret_cast<const PyArrayObject*>(m_obj)), index);
    }
    
    reference& operator[](npy_intp* index)
    {
      return *(pointer)PyArray_GetPtr(reinterpret_cast<PyArrayObject*>(m_obj), index);
      //return *(value_type*)PyArray_GETPTR2(reinterpret_cast<PyArrayObject*>(m_obj), index[0], index[1]);
    }
    
    // assumes 1-D
    template<typename U>
    std::vector<U> toVector() const
    {
      const int n = storageSize();
      std::vector<U> v(n);
      npy_intp i[1]; 
      for (i[0] = 0; i[0] < n; ++i[0])
      {
        v[i[0]] = (U)this->operator[](i);
      }
      return v;
    }
    
    // TODO dimension
    int dim() const 
    {
      return PyArray_NDIM((PyArrayObject*)m_obj);
    }
    
    npy_intp* shape() const 
    {
      return PyArray_SHAPE((PyArrayObject*)m_obj);
    }
    
    // total number of elements
    int storageSize() const
    {
      return PyArray_Size(m_obj);
    }
    
    //npy_intp* shape() const  
    
    long stride(int d)
    {
      return PyArray_STRIDE((PyArrayObject*)m_obj, d);
    }
    
    T* rawData() 
    {
      return (T*)PyArray_DATA((PyArrayObject*)m_obj);
    }
    
  };
  
}


