
#include "Object.h"
#include <Python.h>
#include <stdexcept>


#include <iostream>

PyObject* pycpp::Object::operator&() const 
{
  return m_obj;
}

bool pycpp::Object::operator!() const
{
  return m_obj == nullptr;
}

// This will ensure refcount is at least 1 when returned to python
PyObject* pycpp::Object::release()
{
  Py_INCREF(m_obj);
  return m_obj;
}

pycpp::Object::Object(const pycpp::Object& obj)
{
  Py_INCREF(m_obj);
  m_obj = obj.m_obj;
}

pycpp::Object::Object(pycpp::Object&& obj)
{
  Py_INCREF(m_obj);
  m_obj = obj.m_obj;
}

pycpp::Object& pycpp::Object::operator=(const pycpp::Object& obj)
{
  if (m_obj != obj.m_obj)
  {
    Py_DECREF(m_obj);
    m_obj = obj.m_obj;
    Py_INCREF(m_obj);
  }
  return *this;
}

pycpp::Object::Object(PyObject* obj) : m_obj(obj) 
{ 
  Py_INCREF(m_obj);
  if (m_obj == nullptr)
    throw std::runtime_error("PyObject init failure");
}

pycpp::Object::~Object() 
{ 
  Py_DECREF(m_obj);
}

pycpp::Bool::Bool(bool b) : pycpp::Object(b ? Py_True : Py_False) { }

pycpp::Bool::Bool(PyObject* p) : pycpp::Object(p) 
{ 
  if (!PyBool_Check(m_obj))
    throw std::runtime_error("object is not a bool");
}

pycpp::Bool::operator bool() const 
{
  // this may well not work...
  return m_obj == Py_True;
}

pycpp::Int::Int(int i) : pycpp::Object(PyLong_FromLong(i)) { }

pycpp::Int::Int(uint32_t i) : pycpp::Object(PyLong_FromUnsignedLong(i)) { }

pycpp::Int::Int(int64_t i) : pycpp::Object(PyLong_FromLongLong(i)) { }

pycpp::Int::Int(size_t i) : pycpp::Object(PyLong_FromSize_t(i)) { }

pycpp::Int::Int(PyObject* p) : pycpp::Object(p) 
{ 
  if (!PyLong_Check(m_obj))
    throw std::runtime_error("object is not an int");
}

pycpp::Int::operator int() const 
{
  return PyLong_AsLong(m_obj);
}

pycpp::Int::operator uint32_t() const 
{
  return PyLong_AsLong(m_obj);
}

pycpp::Int::operator size_t() const 
{
  return PyLong_AsUnsignedLongLong(m_obj);
}


pycpp::Double::Double(double x) : pycpp::Object(PyFloat_FromDouble(x)) { }

pycpp::Double::Double(PyObject* p) : pycpp::Object(p) 
{ 
  if (!PyFloat_Check(m_obj))
    throw std::runtime_error("object is not a double");
}

pycpp::Double::operator double() const 
{
  return PyFloat_AsDouble(m_obj);
}


pycpp::String::String(const char* s) : pycpp::Object(PyUnicode_FromString(s)) { }

pycpp::String::String(const std::string& s) : pycpp::Object(PyUnicode_FromString(s.c_str())) { }

pycpp::String::String(PyObject* p) : pycpp::Object(p) 
{ 
  if (!PyUnicode_Check(m_obj))
    throw std::runtime_error("object is not a string");
}

pycpp::String::operator const char*() const 
{
  return PyUnicode_AsUTF8(m_obj);
}

pycpp::List::List(size_t length) : pycpp::Object(PyList_New(length)) { }

pycpp::List::List(PyObject* list) : pycpp::Object((PyObject*)list) 
{ 
  if (!PyList_Check(m_obj))
    throw std::runtime_error("object is not a list");
}

int pycpp::List::size() const
{
  return PyList_Size(m_obj);
}

PyObject* pycpp::List::operator[](size_t i) const
{
  //Py_INCREF(m_obj);
  return PyList_GetItem(m_obj, i);
}

void pycpp::List::set(int index, pycpp::Object&& obj)
{
  PyList_SetItem(m_obj, index, &obj);
}

void pycpp::List::push_back(Object&& obj)
{
  /*int*/PyList_Append(m_obj, &obj);
}

pycpp::Dict::Dict() : pycpp::Object(PyDict_New()) { }

pycpp::Dict::Dict(PyObject* dict) : pycpp::Object((PyObject*)dict) 
{ 
  if (!PyDict_Check(m_obj))
    throw std::runtime_error("object is not a dict");
}

int pycpp::Dict::size() const
{
  return PyDict_Size(m_obj);
}

void pycpp::Dict::clear() const
{
  PyDict_Clear(m_obj);
}

PyObject* pycpp::Dict::operator[](const char* k) const
{
  PyObject* p = PyDict_GetItem(m_obj, &pycpp::String(k));
  Py_INCREF(p);
  return p;
}

// Disabled due to mem leak, use rvalue ref version below
// void pycpp::Dict::insert(const char* k, const pycpp::Object& obj)
// {
//   // takes over mem mgmt
//   PyDict_SetItem(m_obj, &pycpp::String(k), &obj);
//   //Py_INCREF(&obj);
// }

void pycpp::Dict::insert(const char* k, pycpp::Object&& obj)
{
  // takes over mem mgmt
  PyDict_SetItem(m_obj, &pycpp::String(k), &obj);
}






