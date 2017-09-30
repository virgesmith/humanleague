
#include "Object.h"
#include "Array.h"

#include "src/Sobol.h"
#include "src/QIWS.h"
#include "src/GQIWS.h"
#include "src/Integerise.h"
#include "src/IPF.h"
#include "src/QIS.h"
#include "src/QISI.h"

#include "src/QSIPF.h"

#include <Python.h>

#include <vector>
#include <stdexcept>

#include <iostream>


pycpp::List flatten(const size_t pop, const NDArray<uint32_t>& t)
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

// // TODO multidim
// template<typename S>
// void doSolve(pycpp::Dict& result, size_t dims, const std::vector<std::vector<uint32_t>>& m)
// {
//   S qiws(m); 
//   result.insert("conv", pycpp::Bool(qiws.solve()));
//   result.insert("result", flatten(qiws.population(), qiws.result()));
//   result.insert("p-value", pycpp::Double(qiws.pValue().first));
//   result.insert("chiSq", pycpp::Double(qiws.chiSq()));
//   result.insert("pop", pycpp::Int(qiws.population()));
// }

// // TODO merge with above when APIs are consistent
// template<size_t D>
// void doSolveIPF(pycpp::Dict& result, size_t dims, const NDArray<D, double>& seed, const std::vector<std::vector<double>>& m)
// {
//   IPF<D> ipf(seed, m); 
//   result.insert("conv", pycpp::Bool(ipf.conv()));
//   // result.insert("p-value", pycpp::Double(qiws.pValue().first));
//   // result.insert("chiSq", pycpp::Double(qiws.chiSq()));
//   result.insert("pop", pycpp::Double(ipf.population()));
//   // DO THIS LAST BECAUSE ITS DESTRUCTIVE!
//   result.insert("result", pycpp::Array<double>(std::move(const_cast<NDArray<D, double>&>(ipf.result()))));
// }


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

    IPF ipf(seed.toNDArray(), marginals); 
    retval.insert("conv", pycpp::Bool(ipf.conv()));
    // result.insert("p-value", pycpp::Double(qiws.pValue().first));
    // result.insert("chiSq", pycpp::Double(qiws.chiSq()));
    retval.insert("pop", pycpp::Double(ipf.population()));
    // DO THIS LAST BECAUSE ITS DESTRUCTIVE!
    retval.insert("result", pycpp::Array<double>(std::move(const_cast<NDArray<double>&>(ipf.result()))));

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
extern "C" PyObject* humanleague_wip_ipf(PyObject *self, PyObject *args)
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

    wip::IPF<double> ipf(indices, marginals); 
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

    pycpp::Dict result;
    
    QSIPF qsipf(seed.toNDArray(), marginals); 
    result.insert("conv", pycpp::Bool(qsipf.conv()));
    result.insert("chiSq", pycpp::Double(qsipf.chiSq()));
    result.insert("pop", pycpp::Int(qsipf.population()));
    // result.insert("p-value", pycpp::Double(qiws.pValue().first));
    // DO THIS LAST BECAUSE ITS DESTRUCTIVE!
    result.insert("result", pycpp::Array<int64_t>(std::move(const_cast<NDArray<int64_t>&>(qsipf.sample()))));

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
extern "C" PyObject* humanleague_wip_qis(PyObject *self, PyObject *args)
{
  try 
  {
    PyObject* indexArg;
    PyObject* arrayArg;
    
    // args e.g. "s" for string "i" for integer, "d" for float "ss" for 2 strings
    if (!PyArg_ParseTuple(args, "O!O!", &PyList_Type, &indexArg, &PyList_Type, &arrayArg))
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

    wip::QIS qis(indices, marginals); 
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

// // prevents name mangling (but works without this)
// extern "C" PyObject* humanleague_synthPopR(PyObject *self, PyObject *args)
// {
//   try 
//   {
//     PyObject* marginal0Arg;
//     PyObject* marginal1Arg;
//     double rho;

//     // args e.g. "s" for string "i" for integer, "d" for float "ss" for 2 strings
//     if (!PyArg_ParseTuple(args, "O!O!d", &PyList_Type, &marginal0Arg, 
//                                          &PyList_Type, &marginal1Arg, &rho))
//       return nullptr;
      
//     pycpp::List marginal0(marginal0Arg);
//     pycpp::List marginal1(marginal1Arg);
    
//     std::vector<std::vector<uint32_t>> marginals(2);
      
//     marginals[0] = marginal0.toVector<uint32_t>();
//     marginals[1] = marginal1.toVector<uint32_t>();
//     RQIWS rqiws(marginals, rho);
//     pycpp::Dict retval;
//     retval.insert("conv", pycpp::Bool(rqiws.solve()));
//     retval.insert("result", flatten(rqiws.population(), rqiws.result()));
//     retval.insert("pop", pycpp::Int(rqiws.population()));
//     return retval.release();
//   }
//   catch(const std::exception& e)
//   {
//     return &pycpp::String(e.what());
//   }
//   catch(...)
//   {
//     return &pycpp::String("unexpected exception");
//   }
// }

// // prevents name mangling (but works without this)
// extern "C" PyObject* humanleague_numpytest(PyObject *self, PyObject *args)
// {
//   try 
//   {
//     PyObject* arrayArg;

//     // args e.g. "s" for string "i" for integer, "d" for float "ss" for 2 strings
//     if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &arrayArg))
//       return nullptr;
      
//     //npy_intp p[2] = {5,5};
//     //pycpp::Array<double> retval(2, p);
    
//     long lsizes[] = {5,5};
//     size_t sizes[] = {5,5};

//     old::NDArray<2,int64_t> a(sizes);
//     int i = 0;
//     for (old::Index<2, old::Index_Unfixed> idx(sizes); !idx.end(); ++idx)
//     {
//       a[idx] = ++i;
//     }
//     pycpp::Array<int64_t> array(std::move(a));

//     pycpp::Array<int64_t> array2(2, lsizes);
    
//     long index[] = { 0, 0 };
//     for (; index[0] < lsizes[0]; ++index[0])
//       for (; index[1] < lsizes[1]; ++index[1])
//         array2[index] = index[0] * 10 + index[1] * 100;

//     pycpp::Dict retval;
// //    retval.insert("uninit", std::move(array));

//     for (ssize_t i = 0; i < 2; ++i)
//     {
//       std::cout << array2.stride(i) << ", ";
//     }
//     std::cout << std::endl;
//     int64_t* p = array2.rawData();
//     for (ssize_t i = 0; i < array2.storageSize(); ++i)
//     {
//       std::cout << p[i] << ", ";
//       p[i] = i;
//     }
//     std::cout << std::endl;
    
//     retval.insert("init", std::move(array2));
//     retval.insert("NDArray", std::move(array));
    
//     return retval.release();
//   }
//   catch(const std::exception& e)
//   {
//     return &pycpp::String(e.what());
//   }
//   catch(...)
//   {
//     return &pycpp::String("unexpected exception");
//   }
// }

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
  {"wip_ipf", humanleague_wip_ipf, METH_VARARGS, "Synthpop (IPF)."},
  {"qsipf", humanleague_qsipf, METH_VARARGS, "Synthpop (quasirandom sampled IPF)."},
  {"qis", humanleague_wip_qis, METH_VARARGS, "QIS."},
  {"synthPopG", humanleague_synthPopG, METH_VARARGS, "Synthpop generalised."},
  //{"numpytest", humanleague_numpytest, METH_VARARGS, "numpy test."},
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


