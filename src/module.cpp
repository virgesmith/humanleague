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

#define STR2(x) #x
#define STR(x) STR2(x)


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
// explicit array_t(ShapeContainer shape, const T *ptr = nullptr, handle base = handle())
//         : array_t(private_ctor{}, std::move(shape),
//                 ExtraFlags & f_style
//                 ? detail::f_strides(*shape, itemsize())
//                 : detail::c_strides(*shape, itemsize()),
//                 ptr, base) { }


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

py::dict prob2IntFreq(py::array_t<double> frac_a, int pop)
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

  // TODO tuple might be better
  py::dict result;
  result["freq"] = py::array_t<int>(freq.size(), freq.data());
  result["rmse"] = var;

  return result;
}

extern py::array_t<double> sobol(int dim, int length, int skips = 0)
{
  if (dim < 1 || dim > 1111)
  {
    throw py::value_error("Dim %% is not in valid range [1,1111]"_s % dim);
  }

  std::vector<int64_t> sizes{ length, dim };
  py::array_t<double> sequence(sizes);

  Sobol sobol(dim, skips);
  const double scale = 0.5 / (1u << 31);

  for (double* p = begin(sequence); p != end(sequence); ++p)
  {
    *p = sobol() * scale;
  }

  return sequence;
}

py::dict integerise(const py::array_t<double>& npseed)
{
  const NDArray<double> seed = asNDArray<double>(npseed); // shallow copy
  Integeriser integeriser(seed);

  py::dict retval;
  retval["result"] = fromNDArray<int64_t>(integeriser.result());
  retval["conv"] = integeriser.conv();
  retval["rmse"] = integeriser.rmse();
  return retval;
}

py::dict ipf(const py::array_t<double>& seed, const py::list& ilist, const py::list& mlist)
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

  py::dict retval;
  retval["result"] = fromNDArray<double>(result);
  retval["conv"] = ipf.conv();
  retval["pop"] = ipf.population();
  retval["iterations"] = ipf.iters();
  retval["maxError"] = ipf.maxError();

  return retval;
}


py::dict qis(const py::list& ilist, const py::list& mlist, int64_t skips)
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
  py::dict retval;

  retval["result"] = fromNDArray<int64_t>(result);
  retval["expectation"] = fromNDArray<double>(expect);
  retval["conv"] = qis.conv();
  retval["pop"] = qis.population();
  retval["chiSq"] = qis.chiSq();
  retval["pValue"] = qis.pValue();
  retval["degeneracy"] = qis.degeneracy();

  return retval;
}

py::dict qisi(const py::array_t<double> seed, const py::list& ilist, const py::list& mlist, int64_t skips)
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
  const NDArray<double>& expect = qisi.expectation();
  py::dict retval;

  retval["result"] = fromNDArray<int64_t>(result);
  retval["expectation"] = fromNDArray<double>(expect);
  retval["conv"] = qisi.conv();
  retval["pop"] = qisi.population();
  retval["chiSq"] = qisi.chiSq();
  retval["pValue"] = qisi.pValue();
  retval["degeneracy"] = qisi.degeneracy();

  return retval;
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

using namespace py::literals;


PYBIND11_MODULE(humanleague, m) {

#include "docstr.inl"

  m.doc() = module_docstr;

  m.def("version", 
        []() { return STR(HUMANLEAGUE_VERSION); }, 
        version_docstr)
   .def("flatten", 
        hl::flatten, 
        flatten_docstr, 
        "pop"_a)
   .def("prob2IntFreq", 
        hl::prob2IntFreq, 
        prob2IntFreq_docstr, 
        "probs"_a, "pop"_a)
   .def("integerise", 
        hl::prob2IntFreq, 
        prob2IntFreq_docstr, 
        "probs"_a, "pop"_a)
   .def("integerise", 
        hl::integerise, 
        integerise_docstr, 
        "pop"_a)
   .def("sobolSequence", 
        hl::sobol, 
        sobolSequence_docstr, 
        "dim"_a, "length"_a, "skips"_a)
   .def("sobolSequence", 
        [](int dim, int length) { return hl::sobol(dim, length, 0); }, 
        sobolSequence_docstr, 
        "dim"_a, "length"_a)
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
        qis_docstr, 
        "indices"_a, "marginals"_a)
   .def("qisi", 
        hl::qisi, 
        qisi_docstr, 
        "seed"_a, "indices"_a, "marginals"_a, "skips"_a)
   .def("qisi", 
        [](const py::array_t<double>& seed, const py::list& indices, const py::list& marginals) { return hl::qisi(seed, indices, marginals, 0); }, 
        qis_docstr, 
        "seed"_a, "indices"_a, "marginals"_a)
   .def("unittest", 
        hl::unittest)
    ;
}

#endif