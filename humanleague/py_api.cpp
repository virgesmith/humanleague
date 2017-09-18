
#include "Object.h"
#include "Array.h"

#include "src/Sobol.h"
#include "src/RQIWS.h"
#include "src/GQIWS.h"
#include "src/Integerise.h"
#include "src/IPF.h"
#include "src/QSIPF.h"
#include "src/IPF2.h"

#define USE_NEW_NDARRAY

#include <Python.h>

#include <vector>
#include <stdexcept>

#include <iostream>

template<size_t D>
pycpp::List flatten(const size_t pop, const NDArray<D, uint32_t>& t)
{
  print(t.rawData(), t.storageSize(), t.sizes()[1]);
  const std::vector<std::vector<int>>& list = listify<D>(pop, t);
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

// TODO merge with above when APIs are consistent
template<size_t D>
void doSolveIPF(pycpp::Dict& result, size_t dims, const NDArray<D, double>& seed, const std::vector<std::vector<double>>& m)
{
  IPF<D> ipf(seed, m); 
  result.insert("conv", pycpp::Bool(ipf.conv()));
  // result.insert("p-value", pycpp::Double(qiws.pValue().first));
  // result.insert("chiSq", pycpp::Double(qiws.chiSq()));
  result.insert("pop", pycpp::Double(ipf.population()));
  // DO THIS LAST BECAUSE ITS DESTRUCTIVE!
  result.insert("result", pycpp::Array<double>(std::move(const_cast<NDArray<D, double>&>(ipf.result()))));
}

// TODO merge with above when APIs are consistent
template<size_t D>
void doSolveQSIPF(pycpp::Dict& result, size_t dims, const NDArray<D, double>& seed, const std::vector<std::vector<int64_t>>& m)
{
  QSIPF<D> qsipf(seed, m); 
  result.insert("conv", pycpp::Bool(qsipf.conv()));
  result.insert("chiSq", pycpp::Double(qsipf.chiSq()));
  result.insert("pop", pycpp::Int(qsipf.population()));
  // result.insert("p-value", pycpp::Double(qiws.pValue().first));
  // DO THIS LAST BECAUSE ITS DESTRUCTIVE!
  result.insert("result", pycpp::Array<int64_t>(std::move(const_cast<NDArray<D, int64_t>&>(qsipf.sample()))));
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
extern "C" PyObject* humanleague_ipf(PyObject *self, PyObject *args)
{
  try 
  {
    PyObject* arrayArg;
    PyObject* seedArg;
    
    // args e.g. "s" for string "i" for integer, "d" for float "ss" for 2 strings
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, & seedArg, &PyList_Type, &arrayArg))
      return nullptr;
    
    // seed
    pycpp::Array<double> seed(seedArg);
    // expects a list of numpy arrays containing int64
    pycpp::List list(arrayArg);
    
    size_t dim = list.size();
    std::vector<size_t> sizes(dim);
    std::vector<std::vector<double>> marginals(dim);
      
    for (size_t i = 0; i < dim; ++i) 
    {
      if (!PyArray_Check(list[i]))
        throw std::runtime_error("input should be a list of numpy integer arrays");
      pycpp::Array<double> a(list[i]);
      sizes[i] = a.shape()[0];
      marginals[i] = a.toVector<double>();
    }

    pycpp::Dict retval;
    //const NDArray<2, double>& x = seed.toNDArray<2>();

#ifdef USE_NEW_NDARRAY

//void doSolveIPF(pycpp::Dict& result, size_t dims, const NDArray<D, double>& seed, const std::vector<std::vector<double>>& m)
  wip::IPF ipf(seed.toWipNDArray(), marginals); 
  retval.insert("conv", pycpp::Bool(ipf.conv()));
  // result.insert("p-value", pycpp::Double(qiws.pValue().first));
  // result.insert("chiSq", pycpp::Double(qiws.chiSq()));
  retval.insert("pop", pycpp::Double(ipf.population()));
  // DO THIS LAST BECAUSE ITS DESTRUCTIVE!
  retval.insert("result", pycpp::Array<double>(std::move(const_cast<wip::NDArray<double>&>(ipf.result()))));

#else
    switch(dim)
    {
    case 2:
      doSolveIPF<2>(retval, dim, std::move(seed.toNDArray<2>()), marginals);
      break;
    case 3:
      doSolveIPF<3>(retval, dim, std::move(seed.toNDArray<3>()), marginals);
      break;
    case 4:
      doSolveIPF<4>(retval, dim, std::move(seed.toNDArray<4>()), marginals);
      break;
    case 5:
      doSolveIPF<5>(retval, dim, std::move(seed.toNDArray<5>()), marginals);
      break;
    case 6:
      doSolveIPF<6>(retval, dim, std::move(seed.toNDArray<6>()), marginals);
      break;
    case 7:
      doSolveIPF<7>(retval, dim, std::move(seed.toNDArray<7>()), marginals);
      break;
    case 8:
      doSolveIPF<8>(retval, dim, std::move(seed.toNDArray<8>()), marginals);
      break;
    case 9:
      doSolveIPF<9>(retval, dim, std::move(seed.toNDArray<9>()), marginals);
      break;
    case 10:
      doSolveIPF<10>(retval, dim, std::move(seed.toNDArray<10>()), marginals);
      break;
    case 11:
      doSolveIPF<11>(retval, dim, std::move(seed.toNDArray<11>()), marginals);
      break;
    case 12:
      doSolveIPF<12>(retval, dim, std::move(seed.toNDArray<12>()), marginals);
      break;
    default:
      throw std::runtime_error("invalid dimensionality: " + std::to_string(dim));
    }
#endif
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
extern "C" PyObject* humanleague_qsipf(PyObject *self, PyObject *args)
{
  try 
  {
    PyObject* arrayArg;
    PyObject* seedArg;
    
    // args e.g. "s" for string "i" for integer, "d" for float "ss" for 2 strings
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, & seedArg, &PyList_Type, &arrayArg))
      return nullptr;
    
    // seed
    pycpp::Array<double> seed(seedArg);
    // expects a list of numpy arrays containing int64
    pycpp::List list(arrayArg);
    
    size_t dim = list.size();
    std::vector<size_t> sizes(dim);
    std::vector<std::vector<int64_t>> marginals(dim);
      
    for (size_t i = 0; i < dim; ++i) 
    {
      if (!PyArray_Check(list[i]))
        throw std::runtime_error("input should be a list of numpy integer arrays");
      pycpp::Array<int64_t> a(list[i]);
      sizes[i] = a.shape()[0];
      marginals[i] = a.toVector<int64_t>();
    }

    pycpp::Dict retval;
    
    switch(dim)
    {
    case 2:
      doSolveQSIPF<2>(retval, dim, std::move(seed.toNDArray<2>()), marginals);
      break;
    case 3:
      doSolveQSIPF<3>(retval, dim, std::move(seed.toNDArray<3>()), marginals);
      break;
    case 4:
      doSolveQSIPF<4>(retval, dim, std::move(seed.toNDArray<4>()), marginals);
      break;
    case 5:
      doSolveQSIPF<5>(retval, dim, std::move(seed.toNDArray<5>()), marginals);
      break;
    case 6:
      doSolveQSIPF<6>(retval, dim, std::move(seed.toNDArray<6>()), marginals);
      break;
    case 7:
      doSolveQSIPF<7>(retval, dim, std::move(seed.toNDArray<7>()), marginals);
      break;
    case 8:
      doSolveQSIPF<8>(retval, dim, std::move(seed.toNDArray<8>()), marginals);
      break;
    case 9:
      doSolveQSIPF<9>(retval, dim, std::move(seed.toNDArray<9>()), marginals);
      break;
    case 10:
      doSolveQSIPF<10>(retval, dim, std::move(seed.toNDArray<10>()), marginals);
      break;
    case 11:
      doSolveQSIPF<11>(retval, dim, std::move(seed.toNDArray<11>()), marginals);
      break;
    case 12:
      doSolveQSIPF<12>(retval, dim, std::move(seed.toNDArray<12>()), marginals);
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
    pycpp::Array<int64_t> marginal1(marginal1Arg);
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

    NDArray<2,int64_t> a(sizes);
    int i = 0;
    for (Index<2, Index_Unfixed> idx(sizes); !idx.end(); ++idx)
    {
      a[idx] = ++i;
    }
    pycpp::Array<int64_t> array(std::move(a));

    pycpp::Array<int64_t> array2(2, lsizes);
    
    long index[] = { 0, 0 };
    for (; index[0] < lsizes[0]; ++index[0])
      for (; index[1] < lsizes[1]; ++index[1])
        array2[index] = index[0] * 10 + index[1] * 100;

    pycpp::Dict retval;
//    retval.insert("uninit", std::move(array));

    for (ssize_t i = 0; i < 2; ++i)
    {
      std::cout << array2.stride(i) << ", ";
    }
    std::cout << std::endl;
    int64_t* p = array2.rawData();
    for (ssize_t i = 0; i < array2.storageSize(); ++i)
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

// until I find a better way...
extern "C" PyObject* humanleague_version(PyObject*, PyObject*)
{
  static pycpp::Int v(MAJOR_VERSION);
  return v.release();
}

namespace {

// Python2.7
PyMethodDef entryPoints[] = {
  {"prob2IntFreq", humanleague_prob2IntFreq, METH_VARARGS, "Returns nearest-integer population given probs and overall population."},
  {"sobolSequence", humanleague_sobol, METH_VARARGS, "Returns a Sobol sequence."},
  {"synthPop", humanleague_synthPop, METH_VARARGS, "Synthpop."},
  {"ipf", humanleague_ipf, METH_VARARGS, "Synthpop (IPF)."},
  {"qsipf", humanleague_qsipf, METH_VARARGS, "Synthpop (quasirandom sampled IPF)."},
  {"synthPopR", humanleague_synthPopR, METH_VARARGS, "Synthpop correlated."},
  {"synthPopG", humanleague_synthPopG, METH_VARARGS, "Synthpop generalised."},
  {"numpytest", humanleague_numpytest, METH_VARARGS, "numpy test."},
  {"version", humanleague_version, METH_NOARGS, "version info"},
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


