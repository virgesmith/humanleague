// module.cpp - skip if not a python build
#ifdef PYTHON_MODULE

#include "Sobol.h"
#include "Integerise.h"
#include "IPF.h"
#include "QIS.h"
#include "QISI.h"

#include "UnitTester.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using namespace py::literals;


namespace hl {

template<typename T>
T* begin(py::array_t<T>& a)
{
  //assert(a.itemsize() == sizeof(T));
  return (T*)a.request().ptr;
}

template<typename T>
T* end(py::array_t<T>& a)
{
  //assert(a.itemsize() == sizeof(T));
  return (T*)a.request().ptr + a.size();
}

template<typename T>
const T* cbegin(const py::array_t<T>& a)
{
  //assert(a.itemsize() == sizeof(T));
  return (const T*)a.request().ptr;
}

template<typename T>
const T* cend(const py::array_t<T>& a)
{
  //assert(a.itemsize() == sizeof(T));
  return (const T*)a.request().ptr + a.size();
}

template<typename T>
std::vector<T> toVector(const py::array_t<T>& a)
{
  if (a.ndim() != 1)
  {
    throw py::value_error("cannot convert multidimensional array to vector");
  }
  return std::vector<T>(cbegin(a), cend(a));
}

template<typename T>
NDArray<T> toNDArray(const py::array_t<T>& np)
{
  const size_t dim = np.ndim();
  std::vector<int64_t> sizes(dim);
  for (size_t i = 0; i < dim; ++i)
    sizes[i] = np.shape(i);
  NDArray<T> tmp(sizes);
  std::copy(cbegin(np), cend(np), const_cast<T*>(tmp.rawData()));
  return tmp;
}

template<typename T>
NDArray<T> asNDArray(const py::array_t<T>& np)
{
  // this is a bit iffy re: constness
  return NDArray<T>(std::vector<int64_t>(np.shape(), np.shape() + np.ndim()), const_cast<T*>(cbegin(np)));
}

template<typename T>
py::array_t<T> fromNDArray(const NDArray<T>& a)
{
  // TODO ensure this is safe. may need to explicitly copy data
  return py::array_t<T>(a.sizes(), a.rawData());

}

py::list flatten(const py::array_t<int64_t>& a)
{
  const NDArray<int64_t> array = asNDArray<int64_t>(a);

  size_t pop = 0;
  for (Index i(array.sizes()); !i.end(); ++i)
  {
    pop += array[i];
  }

  const std::vector<std::vector<int>>& list = listify(pop, array);
  py::list outer; //array.dim());
  for (size_t i = 0; i < array.dim(); ++i)
  {
    py::list l(list[i].size());
    for(size_t j = 0; j < list[i].size(); ++j) { l[j] = list[i][j]; }
    outer.insert(i, l);
  }

  return outer;
}

py::tuple integerise1d(py::array_t<double> frac_a, int pop)
{
  if (pop < 0)
  {
    throw py::value_error("population cannot be negative");
  }

  // convert py::array_t to vector and normalise it to get probabilities
  std::vector<double> prob = toVector(frac_a);
  double sum = std::accumulate(prob.begin(), prob.end(), 0.0);
  for (double& p: prob) {
    p /= sum;
  }
  double var = 0.0;
  const std::vector<int>& freq = integeriseMarginalDistribution(prob, pop, var);

  py::dict stats;
  stats["rmse"] = var;

  return py::make_tuple(py::array_t<int>(freq.size(), freq.data()), stats);
}


class SobolGenerator
{
public:
  SobolGenerator(uint32_t dim, uint32_t nSkip = 0) : m_sobol(dim, nSkip) { }

  py::array_t<double> next()
  {
    py::array_t<double> sequence(m_sobol.dim());
    try
    {
      const std::vector<uint32_t>& buf = m_sobol.buf();
      std::transform(buf.cbegin(), buf.cend(), begin(sequence), [](uint32_t i) { return i * Sobol::SCALE; });
      return sequence;
    }
    catch(const std::runtime_error&)
    {
      throw py::stop_iteration();
    }
  }

  SobolGenerator& iter()
  {
    return *this;
  }

private:
    Sobol m_sobol;
};



py::tuple integerise(const py::array_t<double>& npseed)
{
  const NDArray<double> seed = asNDArray<double>(npseed); // shallow copy
  Integeriser integeriser(seed);

  py::dict stats(
    "conv"_a = integeriser.conv(),
    "rmse"_a = integeriser.rmse()
  );
  return py::make_tuple(fromNDArray<int64_t>(integeriser.result()), stats);
}

py::tuple ipf(const py::array_t<double>& seed, const py::list& ilist, const py::list& mlist)
{
  size_t k = ilist.size();
  if (k != mlist.size())
    throw py::value_error("index and marginals lists differ in size");
  std::vector<std::vector<int64_t>> indices(k);
  std::vector<NDArray<double>> marginals;
  marginals.reserve(k);

  // TODO assert dtypes are as expected

  for (size_t i = 0; i < k; ++i)
  {
    const py::array_t<int64_t> ia = ilist[i].cast<py::array_t<int64_t>>();
    const py::array_t<double> ma = mlist[i].cast<py::array_t<double>>();
    indices[i] = toVector<int64_t>(ia);
    marginals.push_back(asNDArray<double>(ma));
  }

  IPF<double> ipf(indices, marginals);
  const NDArray<double>& result = ipf.solve(asNDArray<double>(seed));

  py::dict stats(
    "conv"_a = ipf.conv(),
    "pop"_a = ipf.population(),
    "iterations"_a = ipf.iters(),
    "maxError"_a = ipf.maxError()
  );
  return py::make_tuple(fromNDArray<double>(result), stats);
}


py::tuple qis(const py::list& ilist, const py::list& mlist, int64_t skips)
{
  size_t k = ilist.size();
  if (k != mlist.size())
    throw py::value_error("index and marginals lists differ in size");
  std::vector<std::vector<int64_t>> indices(k);
  std::vector<NDArray<int64_t>> marginals;
  marginals.reserve(k);

  // TODO assert dtypes are as expected

  for (size_t i = 0; i < k; ++i)
  {
    const py::array_t<int64_t> ia = ilist[i].cast<py::array_t<int64_t>>();
    const py::array_t<int64_t> ma = mlist[i].cast<py::array_t<int64_t>>();
    indices[i] = toVector<int64_t>(ia);
    marginals.emplace_back(toNDArray<int64_t>(ma));
  }

  QIS qis(indices, marginals, skips);
  const NDArray<int64_t>& result = qis.solve();
  const NDArray<double>& expect = qis.expectation();

  py::dict stats(
    "expectation"_a = fromNDArray<double>(expect),
    "conv"_a = qis.conv(),
    "pop"_a = qis.population(),
    "chiSq"_a = qis.chiSq(),
    "pValue"_a = qis.pValue(),
    "degeneracy"_a = qis.degeneracy()
  );
  return py::make_tuple(fromNDArray<int64_t>(result), stats);
}

py::tuple qisi(const py::array_t<double> seed, const py::list& ilist, const py::list& mlist, int64_t skips)
{
  size_t k = ilist.size();
  if (k != mlist.size())
    throw py::value_error("index and marginals lists differ in size");
  std::vector<std::vector<int64_t>> indices(k);
  std::vector<NDArray<int64_t>> marginals;
  marginals.reserve(k);

  // TODO assert dtypes are as expected

  for (size_t i = 0; i < k; ++i)
  {
    const py::array_t<int64_t> ia = ilist[i].cast<py::array_t<int64_t>>();
    const py::array_t<int64_t> ma = mlist[i].cast<py::array_t<int64_t>>();
    indices[i] = toVector<int64_t>(ia);
    marginals.emplace_back(toNDArray<int64_t>(ma));
  }

  QISI qisi(indices, marginals, skips);
  const NDArray<int64_t>& result = qisi.solve(asNDArray<double>(seed));

  py::dict stats(
    "expectation"_a = fromNDArray<double>(qisi.expectation()),
    "conv"_a = qisi.conv(),
    "pop"_a = qisi.population(),
    "chiSq"_a = qisi.chiSq(),
    "pValue"_a = qisi.pValue(),
    "degeneracy"_a = qisi.degeneracy()
  );
  return py::make_tuple(fromNDArray<int64_t>(result), stats);
}


py::dict unittest()
{
    const unittest::Logger& log = unittest::run();

    py::dict result;
    result["nTests"] = log.testsRun;
    result["nFails"] = log.testsFailed;
    result["errors"] = log.errors;

    return result;
}


} // namespace hl


PYBIND11_MODULE(_humanleague, m) {

#include "docstr.inl"

  m.doc() = module_docstr;

  m.def("flatten",
        hl::flatten,
        flatten_docstr,
        "pop"_a)
   .def("integerise",
        hl::integerise1d,
        integerise1d_docstr,
        "frac"_a, "pop"_a)
   .def("integerise",
        hl::integerise,
        integerise_docstr,
        "pop"_a)
   .def("ipf",
        hl::ipf,
        ipf_docstr,
        "seed"_a, "indices"_a, "marginals"_a)
   .def("qis",
        hl::qis,
        qis_docstr,
        "indices"_a, "marginals"_a, "skips"_a)
   .def("qis",
        [](const py::list& indices, const py::list& marginals) { return hl::qis(indices, marginals, 0); },
        qis2_docstr,
        "indices"_a, "marginals"_a)
   .def("qisi",
        hl::qisi,
        qisi_docstr,
        "seed"_a, "indices"_a, "marginals"_a, "skips"_a)
   .def("qisi",
        [](const py::array_t<double>& seed, const py::list& indices, const py::list& marginals) { return hl::qisi(seed, indices, marginals, 0); },
        qisi2_docstr,
        "seed"_a, "indices"_a, "marginals"_a)
   .def("_unittest",
        hl::unittest,
        unittest_docstr)
    ;

  py::class_<hl::SobolGenerator>(m, "SobolSequence")
      .def(py::init<size_t, uint32_t>(), SobolSequence_init2_docstr, "dim"_a, "skips"_a)
      .def(py::init<size_t>(), SobolSequence_init1_docstr, "dim"_a)
      .def("__iter__", &hl::SobolGenerator::iter, "__iter__ dunder")
      .def("__next__", &hl::SobolGenerator::next, "__next__ dunder")
      ;
}

#endif