
#include "Object.h"
#include "Array.h"

#include "src/Sobol.h"
#include "src/RQIWS.h"
#include "src/GQIWS.h"
#include "src/Integerise.h"

#include <Python.h>

#include <vector>
#include <stdexcept>

#include <iostream>


template<size_t D>
pycpp::List flatten(const size_t pop, const NDArray<D,uint32_t>& t)
{
  std::vector<std::vector<int>> list = listify<D>(pop, t);

  pycpp::List outer(D);
  for (size_t i = 0; i < D; ++i)
  {
//    pycpp::List inner(list[i].size());
//    for (size_t j = 0; j < list[i].size(); ++j) 
//    {
//      inner.set(j, pycpp::Int(list[i][j])); // = pycpp::List(list[i]);    
//    }
//    outer.set(i, std::move(inner));
    outer.set(i, pycpp::List(list[i]));
  }

  return outer;
}

// TODO multidim
template<typename S>
void doSolve(pycpp::Dict& result, size_t dims, const std::vector<std::vector<uint32_t>>& m)
{
  S qiws(m); 
  result.insert("conv", pycpp::Bool(qiws.solve()));
  result.insert("result", flatten(qiws.population(), qiws.result()));
  result.insert("p-value", pycpp::Double(qiws.pValue().first));
  result.insert("chiSq", pycpp::Double(qiws.chiSq()));
  result.insert("pop", pycpp::Int(qiws.population()));
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

    if (fabs(std::accumulate(prob.begin(), prob.end(), -1.0)) > 1000*std::numeric_limits<double>::epsilon())
    {
      throw std::runtime_error("probabilities do not sum to unity");
    }
    std::vector<int> f = integeriseMarginalDistribution(prob, pop, var);

    pycpp::Dict result;
    result.insert("freq", pycpp::Array<int>(f));
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

    size_t sizes[2] = { (size_t)length, (size_t)dim };
    NDArray<2,double> sequence(sizes);

    Sobol sobol(dim, skips);
    double scale = 0.5 / (1u << 31);

    for (Index<2, Index_Unfixed> idx(sizes); !idx.end(); ++idx)
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
      pycpp::List l = pycpp::List(list[i]);
      sizes[i] = l.size();
      marginals[i] = l.toVector<uint32_t>();
    }

    pycpp::Dict retval;

    switch(dim)
    {
    case 2:
      doSolve<QIWS<2>>(retval, dim, marginals);
      break;
    case 3:
      doSolve<QIWS<3>>(retval, dim, marginals);
      break;
    case 4:
      doSolve<QIWS<4>>(retval, dim, marginals);
      break;
    case 5:
      doSolve<QIWS<5>>(retval, dim, marginals);
      break;
    case 6:
      doSolve<QIWS<6>>(retval, dim, marginals);
      break;
    case 7:
      doSolve<QIWS<7>>(retval, dim, marginals);
      break;
    case 8:
      doSolve<QIWS<8>>(retval, dim, marginals);
      break;
    case 9:
      doSolve<QIWS<9>>(retval, dim, marginals);
      break;
    case 10:
      doSolve<QIWS<10>>(retval, dim, marginals);
      break;
    case 11:
      doSolve<QIWS<11>>(retval, dim, marginals);
      break;
    case 12:
      doSolve<QIWS<12>>(retval, dim, marginals);
      break;
    default:
      throw std::runtime_error("invalid dimensionality: " + std::to_string(dim));
    }
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
    pycpp::Array<int> marginal1(marginal1Arg);
    pycpp::Array<double> exoProbs(exoProbsArg);
         
    std::vector<std::vector<uint32_t>> marginals(2);
      
    marginals[0] = marginal0.toVector<uint32_t>();
    marginals[1] = marginal1.toVector<uint32_t>();
    // HACK 
    NDArray<2, double> xp(exoProbs.shape(), exoProbs.rawData());
    GQIWS gqiws(marginals, xp);
    pycpp::Dict retval;
    retval.insert("conv", pycpp::Bool(gqiws.solve()));
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

// prevents name mangling (but works without this)
extern "C" PyObject* humanleague_synthPopR(PyObject *self, PyObject *args)
{
  try 
  {
    PyObject* marginal0Arg;
    PyObject* marginal1Arg;
    double rho;

    // args e.g. "s" for string "i" for integer, "d" for float "ss" for 2 strings
    if (!PyArg_ParseTuple(args, "O!O!d", &PyList_Type, &marginal0Arg, 
                                         &PyList_Type, &marginal1Arg, &rho))
      return nullptr;
      
    pycpp::List marginal0(marginal0Arg);
    pycpp::List marginal1(marginal1Arg);
    
    std::vector<std::vector<uint32_t>> marginals(2);
      
    marginals[0] = marginal0.toVector<uint32_t>();
    marginals[1] = marginal1.toVector<uint32_t>();
    RQIWS rqiws(marginals, rho);
    pycpp::Dict retval;
    retval.insert("conv", pycpp::Bool(rqiws.solve()));
    retval.insert("result", flatten(rqiws.population(), rqiws.result()));
    retval.insert("pop", pycpp::Int(rqiws.population()));
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
extern "C" PyObject* humanleague_numpytest(PyObject *self, PyObject *args)
{
  try 
  {
    PyObject* arrayArg;

    // args e.g. "s" for string "i" for integer, "d" for float "ss" for 2 strings
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &arrayArg))
      return nullptr;
      
    //npy_intp p[2] = {5,5};
    //pycpp::Array<double> retval(2, p);
    
    long lsizes[] = {5,5};
    size_t sizes[] = {5,5};

    NDArray<2,int> a(sizes);
    int i = 0;
    for (Index<2, Index_Unfixed> idx(sizes); !idx.end(); ++idx)
    {
      a[idx] = ++i;
    }
    pycpp::Array<int> array(std::move(a));

    pycpp::Array<int> array2(2, lsizes);
    
    long index[] = { 0, 0 };
    for (; index[0] < lsizes[0]; ++index[0])
      for (; index[1] < lsizes[1]; ++index[1])
        array2[index] = index[0] * 10 + index[1] * 100;

    pycpp::Dict retval;
//    retval.insert("uninit", std::move(array));

    for (int i = 0; i < 2; ++i)
    {
      std::cout << array2.stride(i) << ", ";
    }
    std::cout << std::endl;
    int* p = array2.rawData();
    for (int i = 0; i < array2.storageSize(); ++i)
    {
      std::cout << p[i] << ", ";
      p[i] = i;
    }
    std::cout << std::endl;
    
    retval.insert("init", std::move(array2));
    retval.insert("NDArray", std::move(array));
    
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


namespace {

// Python2.7
PyMethodDef entryPoints[] = {
  {"prob2IntFreq", humanleague_prob2IntFreq, METH_VARARGS, "Returns nearest-integer population given probs and overall population."},
  {"sobolSequence", humanleague_sobol, METH_VARARGS, "Returns a Sobol sequence."},
  {"synthPop", humanleague_synthPop, METH_VARARGS, "Synthpop."},
  {"synthPopR", humanleague_synthPopR, METH_VARARGS, "Synthpop correlated."},
  {"synthPopG", humanleague_synthPopG, METH_VARARGS, "Synthpop generalised."},
  {"numpytest", humanleague_numpytest, METH_VARARGS, "numpy test."},
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


