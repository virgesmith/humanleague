
#pragma once 

#include "Object.h"
#include "humanleague/src/NDArray.h"
#include <Python.h>
#include <numpy/arrayobject.h>

// TODO use boost.python (requires >= 1.63, 16.4 comes with 1.58)
// or https://github.com/ndarray/Boost.NumPy
#include <boost/python/numpy.hpp>

// See https://docs.scipy.org/doc/numpy/reference/c-api.html


namespace pycpp {

  inline void numpy_init()
  {
    import_array();
  }
  
  // Utilities for numpy API
//  // convert size_t into ints that npy understands
//  inline npy_intp* convert(size_t n, const size_t* data)
//  {
//    // LEAKS! FIX
//    npy_intp* p = new npy_intp[n];
//    for (size_t i = 0; i < n; ++i)
//    {
//      p[i] = data[i];
//    }
//    return p;
//  };

  template<typename T> struct NpyType;

  template<> struct NpyType<double> { static const int Type = NPY_DOUBLE; };
  template<> struct NpyType<int> { static const int Type = NPY_INT; };
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

    //explicit Array(PyObject* array);
    
    // Construct from NDArray<D,T>. Data is presumed to be copied
    template<size_t D>
    explicit Array(const NDArray<D, T>& a) 
      : Object(PyArray_SimpleNewFromData(D, convert(D, a.sizes()), NpyType<T>::Type, const_cast<void*>((const void*)a.rawData()))) 
    {
    }
    
    ~Array()
    {
      // do we need to delete/decref?
    }
    
    // generic n-D access??
    const_reference& operator[](npy_intp* index) const
    {
      return *(const_pointer)PyArray_GetPtr(const_cast<PyArrayObject*>(m_obj), index);
    }
    
    reference& operator[](npy_intp* index)
    {
      return *(pointer)PyArray_GetPtr(reinterpret_cast<PyArrayObject*>(m_obj), index);
      //return *(value_type*)PyArray_GETPTR2(reinterpret_cast<PyArrayObject*>(m_obj), index[0], index[1]);
    }
    
    // total number of elements
    int storageSize() const
    {
      return PyArray_Size(m_obj);
    }
    
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


