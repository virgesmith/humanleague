
#pragma once 

#include "Object.h"
#include "humanleague/src/NDArray.h"
#include "humanleague/src/Global.h"
#include <Python.h>
#include <numpy/arrayobject.h>

// See https://docs.scipy.org/doc/numpy/reference/c-api.html

namespace pycpp {

  // Utilities for numpy API
  struct NumPy 
  {
    NumPy() 
    {
      import_array();
    }
    
    ~NumPy()
    {
      // nothing to do here?
    }
    
    // convert size_t into ints that npy understands
    static npy_intp* convert(size_t n, const size_t* data)
    {
      // LEAKS! FIX
      npy_intp* p = new npy_intp[n];
      for (size_t i = 0; i < n; ++i)
      {
        p[i] = data[i];
      }
      return p;
    }
  };

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
    typedef T& reference;
    typedef const T& const_reference;
  
    // uninitialised array with given dimension and sizes
    explicit Array(size_t dim, npy_intp* sizes) 
      : Object(PyArray_SimpleNew(dim, sizes, NpyType<T>::Type)) 
    { }

//    explicit Array(PyObject* array);
    
    // Construct from NDArray<D,T>. Data is presumed to be copied
    template<size_t D>
    explicit Array(const NDArray<D, T>& a) 
      : Object(PyArray_SimpleNewFromData(D, NumPy::convert(D, a.sizes()), NpyType<T>::Type, const_cast<void*>((const void*)a.rawData()))) 
    {
    }
    
    ~Array()
    {
      // do we need to delete/decref?
    }
    
    // generic n-D access??
    const_reference& operator[](npy_intp* index) const
    {
      return *(const T*)PyArray_GetPtr(const_cast<PyArrayObject*>(m_obj), index);
    }
    
    reference& operator[](npy_intp* index)
    {
      //PyArrayObject* p = reinterpret_cast<PyArrayObject*>(m_obj);
      return *(T*)PyArray_GetPtr(reinterpret_cast<PyArrayObject*>(m_obj), index);
    }
    
    // total number of elements
    int size() const
    {
      return PyArray_Size(m_obj);
    }
  };
  
}

// Initialise the API
pycpp::NumPy numpyApi = Global::instance<pycpp::NumPy>(); 

