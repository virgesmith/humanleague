#if 0

#include "Object.h"
#include "Array.h"

#include "src/Sobol.h"
#include "src/Integerise.h"
#include "src/IPF.h"
#include "src/QIS.h"
#include "src/QISI.h"

#include "src/UnitTester.h"

#include <Python.h>

#include <vector>
#include <stdexcept>

#include <iostream>


// deprecated
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

    pycpp::Array<long> pyarray(arrayArg);

    NDArray<int64_t> array(pyarray.toNDArray<int64_t>());

    size_t pop = 0;
    for (Index i(array.sizes()); !i.end(); ++i)
    {
      pop += array[i];
    }

    const std::vector<std::vector<int>>& list = listify(pop, array);
    pycpp::List outer(array.dim());
    for (size_t i = 0; i < array.dim(); ++i)
    {
      outer.set(i, pycpp::List(list[i]));
    }

    return outer.release();
  }
  catch(const std::exception& e)
  {
    return pycpp::String(e.what()).release();
  }
  catch(...)
  {
    return pycpp::String("unexpected exception").release();
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

    if (pop < 0)
    {
      throw std::runtime_error("population cannot be negative");
    }

    if (std::fabs(std::accumulate(prob.begin(), prob.end(), -1.0)) > 1000*std::numeric_limits<double>::epsilon())
    {
      throw std::runtime_error("probabilities do not sum to unity");
    }
    std::vector<int> f = integeriseMarginalDistribution(prob, pop, var);

    pycpp::Dict result;
    result.insert("freq", pycpp::Array<long>(f));
    result.insert("rmse", pycpp::Double(var));

    return result.release();;
  }
  catch(const std::exception& e)
  {
    return pycpp::String(e.what()).release();
  }
  catch(...)
  {
    return pycpp::String("unexpected exception").release();
  }
}

extern "C" PyObject* humanleague_integerise(PyObject *self, PyObject *args)
{
  try
  {
    PyObject* seedArg;

    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, & seedArg))
      return nullptr;

    pycpp::Array<double> npSeed(seedArg);

    NDArray<double> seed = npSeed.toNDArray<double>();
    Integeriser integeriser(seed);

    pycpp::Dict retval;
    retval.insert("result", pycpp::Array<long>(integeriser.result()));
    retval.insert("conv", pycpp::Bool(integeriser.conv()));
    retval.insert("rmse", pycpp::Double(integeriser.rmse()));
    return retval.release();
  }
  catch(const std::exception& e)
  {
    return pycpp::String(e.what()).release();
  }
  catch(...)
  {
    return pycpp::String("unexpected exception").release();
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

    if (dim < 1 || dim > 1111)
      return pycpp::String("Dim %% is not in valid range [1,1111]"_s % dim).release();

    std::vector<int64_t> sizes{ length, dim };
    NDArray<double> sequence(sizes);

    Sobol sobol(dim, skips);
    double scale = 0.5 / (1u << 31);

    for (Index idx(sizes); !idx.end(); ++idx)
    {
      sequence[idx] = sobol() * scale;
    }

    pycpp::Array<double> result(sequence);

    return result.release();
  }
  catch(const std::exception& e)
  {
    return pycpp::String(e.what()).release();
  }
  catch(...)
  {
    return pycpp::String("unexpected exception").release();
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
      pycpp::Array<long> ia(ilist[i]);
      pycpp::Array<double> ma(mlist[i]);
        //sizes[i] = a.shape()[0];
      indices[i] = ia.toVector<int64_t>();
      marginals.push_back(std::move(ma.toNDArray<double>()));
    }

    IPF<double> ipf(indices, marginals);
    const NDArray<double>& result = ipf.solve(seed.toNDArray<double>());

    pycpp::Dict retval;
    retval.insert("result", pycpp::Array<double>(result));
    retval.insert("conv", pycpp::Bool(ipf.conv()));
    retval.insert("pop", pycpp::Double(ipf.population()));
    retval.insert("iterations", pycpp::Int(ipf.iters()));
    // result.insert("errors", ipf.errors());
    retval.insert("maxError", pycpp::Double(ipf.maxError()));

    return retval.release();
  }
  catch(const std::exception& e)
  {
    return pycpp::String(e.what()).release();
  }
  catch(...)
  {
    return pycpp::String("unexpected exception").release();
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
      pycpp::Array<long> ia(ilist[i]);
      pycpp::Array<long> ma(mlist[i]);
        //sizes[i] = a.shape()[0];
      indices[i] = ia.toVector<int64_t>();
      marginals.push_back(std::move(ma.toNDArray<int64_t>()));
    }

    QIS qis(indices, marginals, skips);
    const NDArray<int64_t>& result = qis.solve();
    const NDArray<double>& expect = qis.expectation();
    pycpp::Dict retval;

    retval.insert("result", pycpp::Array<long>(result));
    retval.insert("expectation", pycpp::Array<double>(expect));
    retval.insert("conv", pycpp::Bool(qis.conv()));
    retval.insert("pop", pycpp::Double(qis.population()));
    retval.insert("chiSq", pycpp::Double(qis.chiSq()));
    retval.insert("pValue", pycpp::Double(qis.pValue()));
    retval.insert("degeneracy", pycpp::Double(qis.degeneracy()));

    return retval.release();
  }
  catch(const std::exception& e)
  {
    return pycpp::String(e.what()).release();
  }
  catch(...)
  {
    return pycpp::String("unexpected exception").release();
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
      pycpp::Array<long> ia(ilist[i]);
      pycpp::Array<long> ma(mlist[i]);
        //sizes[i] = a.shape()[0];
      indices[i] = ia.toVector<int64_t>();
      marginals.push_back(std::move(ma.toNDArray<int64_t>()));
    }

    pycpp::Dict retval;

    QISI qisi(indices, marginals, skips);
    retval.insert("result", pycpp::Array<long>(qisi.solve(seed.toNDArray<double>())));
    retval.insert("ipf", pycpp::Array<double>(qisi.expectation()));
    retval.insert("conv", pycpp::Bool(qisi.conv()));
    retval.insert("pop", pycpp::Double(qisi.population()));
    retval.insert("chiSq", pycpp::Double(qisi.chiSq()));
    retval.insert("pValue", pycpp::Double(qisi.pValue()));
    retval.insert("degeneracy", pycpp::Double(qisi.degeneracy()));

    return retval.release();;
  }
  catch(const std::exception& e)
  {
    return pycpp::String(e.what()).release();
  }
  catch(...)
  {
    return pycpp::String("unexpected exception").release();
  }
}


// until I find a better way...
extern "C" PyObject* humanleague_version(PyObject*, PyObject*)
{
  try
  {
    std::string v = std::to_string(MAJOR_VERSION) + "." + std::to_string(MINOR_VERSION) + "." + std::to_string(PATCH_VERSION);
    return pycpp::String(v.c_str()).release();
  }
  catch(const std::exception& e)
  {
    return pycpp::String(e.what()).release();
  }
  catch(...)
  {
    return pycpp::String("unexpected exception").release();
  }
}

extern "C" PyObject* humanleague_unittest(PyObject*, PyObject*)
{
  try
  {
    const unittest::Logger& log = unittest::run();

    pycpp::Dict result;
    result.insert("nTests", pycpp::Int(log.testsRun));
    result.insert("nFails", pycpp::Int(log.testsFailed));
    result.insert("errors", pycpp::List(log.errors));

    return result.release();
  }
  catch(const std::exception& e)
  {
    return pycpp::String(e.what()).release();
  }
  catch(...)
  {
    return pycpp::String("unexpected exception").release();
  }
}

#define APICHECK_EQ(x, y) if (!((x) == (y))) \
  throw std::logic_error(std::string(#x "==" #y "[") + std::to_string(x) + "==" + std::to_string(y) \
    + "] is not true (" + __FILE__ + ":" + std::to_string(__LINE__) + ")");

extern "C" PyObject* humanleague_apitest(PyObject*, PyObject*)
{
  try
  {
    PyObject* p;

    pycpp::Int i0(377027465); // rc=1
    APICHECK_EQ(i0.refcount(), 1);
    {
      pycpp::Int i1(i0);  // rc=2
      APICHECK_EQ(i0.refcount(), 2);
      APICHECK_EQ(i1.refcount(), 2);
    }
    APICHECK_EQ(i0.refcount(), 1);

    p = []() {
      pycpp::List l(1);
      l.set(0, pycpp::Int(12347543));
      APICHECK_EQ(l.refcount(), 1);
      //APICHECK_EQ(Py_REFCNT(l[0]), 3); // leak?
      return l.release();
    }();
    APICHECK_EQ(Py_REFCNT(p), 1);
    APICHECK_EQ(Py_REFCNT(PyList_GetItem(p, 0)), 1);

    p = []() {
      pycpp::Dict d;
      d.insert("key0", pycpp::Int(12375443));
      APICHECK_EQ(d.refcount(), 1);
      APICHECK_EQ(Py_REFCNT(d["key0"]), 1);
      return d.release();
    }();
    APICHECK_EQ(Py_REFCNT(p), 1);
    APICHECK_EQ(Py_REFCNT(PyDict_GetItem(p, &pycpp::String("key0"))), 1);

    {
      npy_intp s[] = { 3, 5 };
      pycpp::Array<double> a(2, s);
      p = &a;
      APICHECK_EQ(a.refcount(), 1);
    }
    //APICHECK_EQ(Py_REFCNT(p), 0); // fails, but probably not a problem (memory could have been overwritten)


    Py_RETURN_NONE;
  }
  catch(const std::exception& e)
  {
    return pycpp::String(e.what()).release();
  }
  catch(...)
  {
    return pycpp::String("unexpected exception").release();
  }
}

namespace {

#include "docstr.inl"

PyMethodDef entryPoints[] = {
  {"prob2IntFreq", humanleague_prob2IntFreq, METH_VARARGS, "Returns nearest-integer population given probs and overall population."},
  {"integerise", humanleague_integerise, METH_VARARGS, "Returns mulidimensional nearest-integer population constrained to marginal sums in every dimension."},
  {"flatten", humanleague_flatten, METH_VARARGS, "Converts n-D integer array into a table with columns referencing the value indices."},
  {"sobolSequence", humanleague_sobol, METH_VARARGS, "Returns a Sobol sequence given a dimension and a length."},
  {"ipf", humanleague_ipf, METH_VARARGS, "Iterative proportional fitting, given a seed population and marginal distributions."},
  {"qis", humanleague_qis, METH_VARARGS, "Quasirandom integer (unweighted) sampling of population given marginal distributions."},
  {"qisi", humanleague_qisi, METH_VARARGS, "Quasirandom integer (weighted) sampling, of population given marginal distributions, using IPF to update the sample distribution."},
  {"version", humanleague_version, METH_NOARGS, version_docstr },
  {"unittest", humanleague_unittest, METH_NOARGS, "run unit tests"},
  {"apitest", humanleague_apitest, METH_NOARGS, "run api (memory) tests"},
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

#endif
