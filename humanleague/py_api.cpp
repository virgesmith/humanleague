
#include "Object.h"
#include "Array.h"

#include "src/Sobol.h"
#include "src/QIWS.h"
#include "src/GQIWS.h"
#include "src/Integerise.h"
#include "src/IPF.h"
#include "src/QIS.h"
#include "src/QISI.h"

#include "src/UnitTester.h"

#include <Python.h>

#include <vector>
#include <stdexcept>

#include <iostream>

template<typename T>
pycpp::List flatten(const size_t pop, const NDArray<T>& t)
{
  //print(t.rawData(), t.storageSize(), t.sizes()[1]);
  const std::vector<std::vector<int>>& list = listify(pop, t);
  pycpp::List outer(t.dim());
  for (size_t i = 0; i < t.dim(); ++i)
  {
    outer.set(i, pycpp::List(list[i]));
  }
  return outer;
}

// flatten n-D integer array into 2-d table
extern "C" PyObject* humanleague_flatten(PyObject* self, PyObject* args)
{
  try 
  {
    PyObject* arrayArg;

    // args e.g. "s" for string "i" for integer, "d" for float "ss" for 2 strings
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &arrayArg))
      return nullptr; 

    pycpp::Array<int64_t> pyarray(arrayArg);
    
    NDArray<int64_t> array(pyarray.toNDArray());

    size_t pop = 0;
    for (Index i(array.sizes()); !i.end(); ++i)
    {
      pop += array[i];
    }

    return flatten(pop, array).release();    
  }
  catch(const std::exception& e)
  {
    return &pycpp::String(e.what());
  }
  catch(...)
  {
    return &pycpp::String("unexpected exception");
  }
}

extern "C" PyObject* humanleague_prob2IntFreq(PyObject* self, PyObject* args)
{
  try 
  {
    PyObject* probArg;
    int pop;

    // args e.g. "s" for string "i" for integer, "d" for float "ss" for 2 strings
    if (!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &probArg, &pop))
      return nullptr;
      
    const std::vector<double>& prob = pycpp::Array<double>(probArg).toVector<double>();

    double var;

    if (pop < 1)
    {
      throw std::runtime_error("population must be strictly positive");
    }

    if (std::fabs(std::accumulate(prob.begin(), prob.end(), -1.0)) > 1000*std::numeric_limits<double>::epsilon())
    {
      throw std::runtime_error("probabilities do not sum to unity");
    }
    std::vector<int> f = integeriseMarginalDistribution(prob, pop, var);

    pycpp::Dict result;
    result.insert("freq", pycpp::Array<int64_t>(f));
    result.insert("var", pycpp::Double(var));

    return result.release();
  }
  catch(const std::exception& e)
  {
    return &pycpp::String(e.what());
  }
  catch(...)
  {
    return &pycpp::String("unexpected exception");
  }
}

// prevents name mangling (but works without this)
extern "C" PyObject* humanleague_sobol(PyObject *self, PyObject *args)
{
  try 
  {
    int dim, length, skips = 0;

    // args e.g. "s" for string "i" for integer, "d" for double "ss" for 2 strings
    if (!PyArg_ParseTuple(args, "ii|i", &dim, &length, &skips))
      return nullptr;

    std::vector<int64_t> sizes{ length, dim };
    NDArray<double> sequence(sizes);

    Sobol sobol(dim, skips);
    double scale = 0.5 / (1u << 31);

    for (Index idx(sizes); !idx.end(); ++idx)
    {
      sequence[idx] = sobol() * scale;
    }
    
    pycpp::Array<double> result(std::move(sequence));
    
    return result.release();
  }
  catch(const std::exception& e)
  {
    return &pycpp::String(e.what());
  }
  catch(...)
  {
    return &pycpp::String("unexpected exception");
  }
}

// prevents name mangling (but works without this)
extern "C" PyObject* humanleague_ipf(PyObject *self, PyObject *args)
{
  try 
  {
    PyObject* indexArg;
    PyObject* arrayArg;
    PyObject* seedArg;
    
    // args e.g. "s" for string "i" for integer, "d" for float "ss" for 2 strings
    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, & seedArg, &PyList_Type, &indexArg, &PyList_Type, &arrayArg))
      return nullptr;
    
    // seed
    pycpp::Array<double> seed(seedArg);
    // expects a list of numpy arrays containing int64
    pycpp::List ilist(indexArg);
    pycpp::List mlist(arrayArg);

    int64_t k = ilist.size();
    if (k != mlist.size())
      throw std::runtime_error("index and marginals lists differ in size");
    //std::vector<size_t> sizes(k);
    std::vector<std::vector<int64_t>> indices(k);
    std::vector<NDArray<double>> marginals;
    marginals.reserve(k);
    
    for (int64_t i = 0; i < k; ++i) 
    {
      if (!PyArray_Check(ilist[i]))
        throw std::runtime_error("index input should be a list of numpy integer arrays");
      if (!PyArray_Check(mlist[i]))
        throw std::runtime_error("marginal input should be a list of numpy float arrays");
      pycpp::Array<int64_t> ia(ilist[i]);
      pycpp::Array<double> ma(mlist[i]);
        //sizes[i] = a.shape()[0];
      indices[i] = ia.toVector<int64_t>();
      marginals.push_back(std::move(ma.toNDArray/*<double>*/()));
    }

    pycpp::Dict retval;

    IPF<double> ipf(indices, marginals); 
    // THIS IS DESTRUCTIVE!
    retval.insert("result", pycpp::Array<double>(std::move(const_cast<NDArray<double>&>(ipf.solve(seed.toNDArray())))));
    retval.insert("conv", pycpp::Bool(ipf.conv()));
    retval.insert("pop", pycpp::Double(ipf.population()));
    retval.insert("iterations", pycpp::Int(ipf.iters()));
    // result.insert("errors", ipf.errors());
    retval.insert("maxError", pycpp::Double(ipf.maxError()));
      
    return retval.release();
  }
  catch(const std::exception& e)
  {
    return &pycpp::String(e.what());
  }
  catch(...)
  {
    return &pycpp::String("unexpected exception");
  }
}


// prevents name mangling (but works without this)
extern "C" PyObject* humanleague_qis(PyObject *self, PyObject *args)
{
  try 
  {
    PyObject* indexArg;
    PyObject* arrayArg;
    int64_t skips = 0;
    
    // args e.g. "s" for string "i" for integer, "d" for float "ss" for 2 strings
    if (!PyArg_ParseTuple(args, "O!O!|i", &PyList_Type, &indexArg, &PyList_Type, &arrayArg, &skips))
      return nullptr;
    
    // seed
    //pycpp::Array<double> seed(seedArg);
    // expects a list of numpy arrays containing int64
    pycpp::List ilist(indexArg);
    pycpp::List mlist(arrayArg);

    int64_t k = ilist.size();
    if (k != mlist.size())
      throw std::runtime_error("index and marginals lists differ in size");
    //std::vector<size_t> sizes(k);
    std::vector<std::vector<int64_t>> indices(k);
    std::vector<NDArray<int64_t>> marginals;
    marginals.reserve(k);
    
    for (int64_t i = 0; i < k; ++i) 
    {
      if (!PyArray_Check(ilist[i]))
        throw std::runtime_error("index input should be a list of numpy integer arrays");
      if (!PyArray_Check(mlist[i]))
        throw std::runtime_error("marginal input should be a list of numpy float arrays");
      pycpp::Array<int64_t> ia(ilist[i]);
      pycpp::Array<int64_t> ma(mlist[i]);
        //sizes[i] = a.shape()[0];
      indices[i] = ia.toVector<int64_t>();
      marginals.push_back(std::move(ma.toNDArray()));
    }

    pycpp::Dict retval;

    QIS qis(indices, marginals, skips); 
    // THIS IS DESTRUCTIVE!
    retval.insert("result", pycpp::Array<int64_t>(std::move(const_cast<NDArray<int64_t>&>(qis.solve()))));
    retval.insert("conv", pycpp::Bool(qis.conv()));
    retval.insert("pop", pycpp::Double(qis.population()));
    retval.insert("chiSq", pycpp::Double(qis.chiSq()));
    retval.insert("pValue", pycpp::Double(qis.pValue()));
    retval.insert("degeneracy", pycpp::Double(qis.degeneracy()));
    
    return retval.release();
  }
  catch(const std::exception& e)
  {
    return &pycpp::String(e.what());
  }
  catch(...)
  {
    return &pycpp::String("unexpected exception");
  }
}

// prevents name mangling (but works without this)
extern "C" PyObject* humanleague_qisi(PyObject *self, PyObject *args)
{
  try 
  {
    PyObject* seedArg;
    PyObject* indexArg;
    PyObject* arrayArg;
    int64_t skips = 0;
    
    // args e.g. "s" for string "i" for integer, "d" for float "ss" for 2 strings
    if (!PyArg_ParseTuple(args, "O!O!O!|i", &PyArray_Type, & seedArg, &PyList_Type, &indexArg, &PyList_Type, &arrayArg, &skips))
      return nullptr;
  
    // seed
    pycpp::Array<double> seed(seedArg);
    //expects a list of numpy arrays containing int64
    pycpp::List ilist(indexArg);
    pycpp::List mlist(arrayArg);

    int64_t k = ilist.size();
    if (k != mlist.size())
      throw std::runtime_error("index and marginals lists differ in size");
    //std::vector<size_t> sizes(k);
    std::vector<std::vector<int64_t>> indices(k);
    std::vector<NDArray<int64_t>> marginals;
    marginals.reserve(k);
    
    for (int64_t i = 0; i < k; ++i) 
    {
      if (!PyArray_Check(ilist[i]))
        throw std::runtime_error("index input should be a list of numpy integer arrays");
      if (!PyArray_Check(mlist[i]))
        throw std::runtime_error("marginal input should be a list of numpy float arrays");
      pycpp::Array<int64_t> ia(ilist[i]);
      pycpp::Array<int64_t> ma(mlist[i]);
        //sizes[i] = a.shape()[0];
      indices[i] = ia.toVector<int64_t>();
      marginals.push_back(std::move(ma.toNDArray()));
    }

    pycpp::Dict retval;

    QISI qisi(indices, marginals, skips);
    // THIS IS DESTRUCTIVE!
    retval.insert("result", pycpp::Array<int64_t>(std::move(const_cast<NDArray<int64_t>&>(qisi.solve(seed.toNDArray())))));
    retval.insert("conv", pycpp::Bool(qisi.conv()));
    retval.insert("pop", pycpp::Double(qisi.population()));
    retval.insert("chiSq", pycpp::Double(qisi.chiSq()));
    retval.insert("pValue", pycpp::Double(qisi.pValue()));
    retval.insert("degeneracy", pycpp::Double(qisi.degeneracy()));
    
    return retval.release();
  }
  catch(const std::exception& e)
  {
    return &pycpp::String(e.what());
  }
  catch(...)
  {
    return &pycpp::String("unexpected exception");
  }
}


// prevents name mangling (but works without this)
extern "C" PyObject* humanleague_synthPop(PyObject *self, PyObject *args)
{
  try 
  {
    PyObject* arrayArg;

    // args e.g. "s" for string "i" for integer, "d" for float "ss" for 2 strings
    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &arrayArg))
      return nullptr;
    
    // expects a list of numpy arrays containing int64
    pycpp::List list(arrayArg);
    
    size_t dim = list.size();
    std::vector<size_t> sizes(dim);
    std::vector<std::vector<uint32_t>> marginals(dim);
      
    for (size_t i = 0; i < dim; ++i) 
    {
      if (!PyArray_Check(list[i]))
        throw std::runtime_error("input should be a list of numpy integer arrays");
      pycpp::Array<int64_t> a = pycpp::Array<int64_t>(list[i]);
      sizes[i] = a.shape()[0];
      marginals[i] = a.toVector<uint32_t>();
    }

    pycpp::Dict retval;
    QIWS qiws(marginals); 
    retval.insert("conv", pycpp::Bool(qiws.solve()));
    // cannot easily output uint32_t array...
    retval.insert("result", flatten(qiws.population(), qiws.result()));
    retval.insert("p-value", pycpp::Double(qiws.pValue().first));
    retval.insert("chiSq", pycpp::Double(qiws.chiSq()));
    retval.insert("pop", pycpp::Int(qiws.population()));
    return retval.release();
  }
  catch(const std::exception& e)
  {
    return &pycpp::String(e.what());
  }
  catch(...)
  {
    return &pycpp::String("unexpected exception");
  }
}

// prevents name mangling (but works without this)
extern "C" PyObject* humanleague_synthPopG(PyObject *self, PyObject *args)
{
  try 
  {
    PyObject* marginal0Arg;
    PyObject* marginal1Arg;
    PyObject* exoProbsArg;

    // args e.g. "s" for string "i" for integer, "d" for float "ss" for 2 strings
    if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &marginal0Arg, &PyArray_Type, &marginal1Arg, &PyArray_Type, &exoProbsArg))
      return nullptr;
      
    pycpp::Array<int64_t> marginal0(marginal0Arg);
    pycpp::Array<int64_t> marginal1(marginal1Arg);
    pycpp::Array<double> exoProbs(exoProbsArg);
         
    std::vector<std::vector<uint32_t>> marginals(2);
      
    marginals[0] = marginal0.toVector<uint32_t>();
    marginals[1] = marginal1.toVector<uint32_t>();
    
    // Borrow memory from the numpy array
    std::vector<int64_t> shape(exoProbs.shape(), exoProbs.shape() + exoProbs.dim());
    NDArray<double> xp(shape, exoProbs.rawData());

    GQIWS gqiws(marginals, xp);
    pycpp::Dict retval;
    retval.insert("conv", pycpp::Bool(gqiws.solve()));
    // cannot easily output uint32_t array...
    retval.insert("result", flatten(gqiws.population(), gqiws.result()));
    //retval.insert("result", pycpp::Array<uint32_t>(std::move(const_cast<NDArray<2,uint32_t>&>(gqiws.result()))));
    retval.insert("pop", pycpp::Int(gqiws.population()));
    return retval.release();
  }
  catch(const std::exception& e)
  {
    return &pycpp::String(e.what());
  }
  catch(...)
  {
    return &pycpp::String("unexpected exception");
  }
}


// until I find a better way...
extern "C" PyObject* humanleague_version(PyObject*, PyObject*)
{
  static pycpp::Int v(MAJOR_VERSION);
  return v.release();
}

extern "C" PyObject* humanleague_unittest(PyObject*, PyObject*)
{
  const unittest::Logger& log = unittest::run();

  pycpp::Dict result;
  result.insert("nTests", pycpp::Int(log.testsRun));
  result.insert("nFails", pycpp::Int(log.testsFailed));
  result.insert("errors", pycpp::List(log.errors));

  return result.release();
}

namespace {

// Python2.7
PyMethodDef entryPoints[] = {
  {"prob2IntFreq", humanleague_prob2IntFreq, METH_VARARGS, "Returns nearest-integer population given probs and overall population."},
  {"flatten", humanleague_flatten, METH_VARARGS, "Converts n-D integer array into a table with columns referencing the value indices."},
  {"sobolSequence", humanleague_sobol, METH_VARARGS, "Returns a Sobol sequence."},
  {"ipf", humanleague_ipf, METH_VARARGS, "Synthpop (IPF)."},
  {"qis", humanleague_qis, METH_VARARGS, "QIS."},
  {"qisi", humanleague_qisi, METH_VARARGS, "QIS-IPF."},
  {"synthPop", humanleague_synthPop, METH_VARARGS, "Synthpop."},
  {"synthPopG", humanleague_synthPopG, METH_VARARGS, "Synthpop generalised."},
  {"version", humanleague_version, METH_NOARGS, "version info"},
  {"unittest", humanleague_unittest, METH_NOARGS, "unit testing"},
  {nullptr, nullptr, 0, nullptr}        /* terminator */
};

PyModuleDef moduleDef =
{
  PyModuleDef_HEAD_INIT,
  "humanleague", /* name of module */
  "",          /* module documentation, may be NULL */
  -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
  entryPoints
};

PyObject *error;

}

// From http://orb.essex.ac.uk/ce/ce705/python-3.5.1-docs-html/extending/extending.html
// This function leaks 161k, obvious candidates are module and error but its more complicated than this
// since removing error completely, and decreffing module doesnt help.
// But this seems to be the officially sanctioned way of doing things, and you want the module's state to 
// persist, so it's probably ok.
PyMODINIT_FUNC PyInit_humanleague()
{
  PyObject *module = PyModule_Create(&moduleDef);

  if (module == nullptr)
    return nullptr;
    
  error = PyErr_NewException("humanleague.Error", nullptr, nullptr);
  Py_INCREF(error);
  PyModule_AddObject(module, "error", error);

  pycpp::numpy_init();
  return module;
}


