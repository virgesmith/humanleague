
#pragma once

//#include <Python.h>
//#include <stdexcept>

#include <string>
#include <vector>
#include <map>
#include <cstddef>

// TODO fix fwd decl
typedef struct _object PyObject;
//typedef struct _list PyList;

namespace pycpp {

  // not directly instantiated, use derived types
  class Object
  {
  public:
  
    PyObject* operator&() const ;
    bool operator!() const;
    
    // Memory mgmt is now callers responsibility
    PyObject* release();
    
  protected:
  
    Object(const Object& obj);
    
    Object& operator=(const Object& obj);
  
    explicit Object(PyObject* obj);

    virtual ~Object();

    PyObject* m_obj; 
  };

  // map C++ scalar type(s) to Object derived type
  template<typename T> struct Type;
  
  class Bool : public Object 
  {
  public:
    explicit Bool(bool x);
    
    explicit Bool(PyObject* p);
    
    operator bool() const;
  };

  template<> struct Type<bool> { typedef Bool PyType; };

  class Int : public Object 
  {
  public:
    explicit Int(int x);
    
    explicit Int(PyObject* p);
    
    operator int() const;
    // NB python doesnt have native unsigned type
    operator uint32_t() const;
    operator size_t() const;
  };
  
  // define C++ types that map to Int
  template<> struct Type<int> { typedef Int PyType; };
  template<> struct Type<uint32_t> { typedef Int PyType; };
  template<> struct Type<size_t> { typedef Int PyType; };

  class Double : public Object 
  {
  public:
    explicit Double(double x);

    explicit Double(PyObject* p);
    
    operator double() const;
  };
  
  template<> struct Type<double> { typedef Double PyType; };

  class String : public Object
  {
  public: 
    explicit String(const char* s);
    explicit String(PyObject* p);

    operator const char*() const; 
    operator std::string() const; 
  };
  
  template<> struct Type<const char*> { typedef String PyType; };
  template<> struct Type<std::string> { typedef String PyType; };

  class List : public Object
  {
  public:
    // uninitialised list with given length
    explicit List(size_t length = 0);
    
    explicit List(PyObject* list);
    
    // Construct from vector of scalar types that correspond to a pycpp type
    template<typename T>
    explicit List(const std::vector<T>& v) : List(v.size())
    {
      for (size_t i = 0; i < v.size(); ++i)
      {
        set(i, typename Type<T>::PyType(v[i]));
      }
    }
    
    // return value given we don't know the actual type??
    PyObject* operator[](size_t i) const;
    
    // set (move semantics)
    void set(int index, Object&& obj);
    
    // append (move semantics)
    void push(Object&& obj);

    // return value given we don't know the actual type??
    //PyObject* get(int index) const;
    
    template<typename T/*, typename P = typename Type<T>::PyType*/>
    std::vector<T> toVector() const
    {
      const size_t n = size();
      std::vector<uint32_t> v(n);
      for (size_t i = 0; i < n; ++i)
      {
        v[i] = typename Type<T>::PyType(this->operator[](i));
      }
      return v;
    }
    
    // appears to be no clear mechanism for lists
    //void clear() const;

    int size() const;
  };
  
  class Dict : public Object
  {
  public:
    // empty dict
    Dict();
    
    explicit Dict(PyObject* dict);
    
    // Construct from map of string to scalar types that correspond to a pycpp type
    template<typename T>
    explicit Dict(const std::map<std::string, T>& m) : Dict()
    {
      for (auto it = m.cbegin(); it != m.cend(); ++it)
      {
        insert(it->first.c_str(), typename Type<T>::PyType(it->second));
      }
    }

    // return value given we don't know the actual type??
    PyObject* operator[](const char*) const;
    
    // set (move semantics)
    void insert(const char*, Object&& obj);
    
    // TODO find...
    
    void clear() const;

    int size() const;
  };
  
}
