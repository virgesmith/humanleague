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
  return NDArray<T>(std::vector<int64_t>(np.shape(), np.shape() + np.ndim()), const_cast<double*>(cbegin(np)));
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


// TODO -> integerise
py::dict prob2IntFreq(py::array_t<double> prob_a, int pop)
{
  if (pop < 0)
  {
    throw py::value_error("population cannot be negative");
  }

  // convert py::array_t to vector
  const std::vector<double> prob(cbegin(prob_a), cend(prob_a));
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

  m.def("version", []() { return STR(HUMANLEAGUE_VERSION); }, version_docstr)
   .def("prob2IntFreq", hl::prob2IntFreq, prob2IntFreq_docstr, "probs"_a, "pop"_a)
   .def("integerise", hl::prob2IntFreq, prob2IntFreq_docstr, "probs"_a, "pop"_a)
   .def("integerise", hl::integerise, integerise_docstr, "pop"_a)
   .def("sobolSequence", hl::sobol, sobolSequence_docstr, "dim"_a, "length"_a, "skips"_a)
   .def("sobolSequence", [](int dim, int length) { return hl::sobol(dim, length, 0); }, sobolSequence_docstr, "dim"_a, "length"_a)
   .def("unittest", hl::unittest);
  ;
}


#endif