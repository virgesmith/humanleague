// module.cpp - skip if not a python build
#ifdef PYTHON_MODULE

#include "IPF.h"
#include "Integerise.h"
#include "QIS.h"
#include "QISI.h"
#include "Sobol.h"

#include "UnitTester.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

using namespace nb::literals;

template <typename... Args> using np_array = nb::ndarray<nb::numpy, Args...>;

namespace hl {

template <typename T> T* begin(np_array<T>& a) {
  // assert(a.itemsize() == sizeof(T));
  return (T*)a.data();
}

template <typename T> T* end(np_array<T>& a) {
  // assert(a.itemsize() == sizeof(T));
  return (T*)a.data() + a.size();
}

template <typename T, typename ...Args> const T* cbegin(const np_array<T, Args...>& a) {
  // assert(a.itemsize() == sizeof(T));
  return (const T*)a.data();
}

template <typename T, typename ...Args> const T* cend(const np_array<T, Args...>& a) {
  // assert(a.itemsize() == sizeof(T));
  return (const T*)a.data() + a.size();
}

template <typename T, typename ...Args> std::vector<T> toVector(const np_array<T, Args...>& a) {
  if (a.ndim() == 0) {
    return std::vector<T>(1, *cbegin(a));
  }
  if (a.ndim() != 1) {
    throw nb::value_error(("cannot convert %%-dimensional array to vector"s % a.ndim()).c_str());
  }
  return std::vector<T>(cbegin(a), cend(a));
}

template <typename T, typename ...Args> NDArray<T> toNDArray(const np_array<T, Args...>& np) {
  const size_t dim = np.ndim();
  std::vector<int64_t> sizes(dim);
  for (size_t i = 0; i < dim; ++i)
    sizes[i] = np.shape(i);
  NDArray<T> tmp(sizes);
  std::copy(cbegin(np), cend(np), const_cast<T*>(tmp.rawData()));
  return tmp;
}

template <typename T, typename ...Args> NDArray<T> asNDArray(const np_array<T, Args...>& np) {
  // this is a bit iffy re: constness
  return NDArray<T>(std::vector<int64_t>(np.shape_ptr(), np.shape_ptr() + np.ndim()), const_cast<T*>(cbegin(np)));
}

template <typename T> py::array_t<T> fromNDArray(const NDArray<T>& a) {
  py::array_t<T> result(a.sizes());
  std::copy(a.rawData(), a.rawData() + a.storageSize(), result.mutable_data());
  return result;
}

std::vector<std::vector<int64_t>> collect_indices(const nb::iterable& iterable) {
  std::vector<std::vector<int64_t>> indices;

  for (const auto& elem : iterable) {
    if (nb::isinstance<nb::int_>(elem)) {
      indices.push_back(std::vector<int64_t>(1, nb::cast<int64_t>(elem)));
    } else if (nb::isinstance<nb::iterable>(elem)) {
      // TODO must be an easier way???
      std::vector<int64_t> index;
      for (const auto& item: elem) {
        index.push_back(nb::cast<int64_t>(item));
      }
      indices.push_back(index);
    } else {
      throw nb::value_error("unexpected type for index");
    }
  }
  return indices;
}

template <typename T> std::vector<NDArray<T>> collect_marginals(const nb::iterable& iterable) {
  std::vector<NDArray<T>> marginals;

  for (const auto& elem : iterable) {
    const np_array<T> ma = nb::cast<np_array<T>>(elem);
    marginals.emplace_back(toNDArray<T>(ma));
  }
  return marginals;
}

nb::list flatten(const np_array<int64_t>& a) {

  auto warnings = nb::module_::import_("warnings");
  warnings.attr("warn")("humanleague.flatten is deprecated, consider using humanleague.tabulate_individuals instead.");

  const NDArray<int64_t> array = asNDArray<int64_t>(a);

  size_t pop = 0;
  for (Index i(array.sizes()); !i.end(); ++i) {
    pop += array[i];
  }

  const std::vector<std::vector<int>>& list = listify(pop, array);
  nb::list outer; // array.dim());
  for (size_t i = 0; i < array.dim(); ++i) {
    nb::list l;
    for (size_t j = 0; j < list[i].size(); ++j) {
      l.append(list[i][j]);
    }
    outer.insert(i, l);
  }

  return outer;
}

nb::tuple integerise1d(np_array<double, nb::ro> frac_a, int pop) {
  if (pop < 0) {
    throw nb::value_error("population cannot be negative");
  }

  // convert np_array to vector and normalise it to get probabilities
  std::vector<double> prob = toVector(frac_a);

  // ensure all values are finite (negative values are permitted)
  for (auto x : prob) {
    if (!std::isfinite(x)) {
      throw py::value_error("Invalid value in input: %%"s % x);
    }
  }

  double sum = std::accumulate(prob.begin(), prob.end(), 0.0);
  for (double& p : prob) {
    p /= sum;
  }
  double var = 0.0;
  const std::vector<int>& freq = integeriseMarginalDistribution(prob, pop, var);

  nb::dict stats;
  stats["conv"] = true; // always converges, but including for consistency
  stats["rmse"] = var;

  return nb::make_tuple(np_array<int, nb::ro>(freq.data(), {freq.size()}), stats);
}

class SobolGenerator {
public:
  SobolGenerator(size_t dim, size_t nSkip = 0) : m_sobol(dim, nSkip) {}

  np_array<nb::numpy, double> next() {
    double* data = new double[m_sobol.dim()];
    // Delete 'data' when the 'owner' capsule expires
    nb::capsule owner(data, [](void* p) noexcept { delete[] (float*)p; });

    try {
      const std::vector<uint32_t>& buf = m_sobol.buf();
      std::transform(buf.cbegin(), buf.cend(), data, [](uint64_t i) { return i * Sobol::SCALE; });
      return np_array<nb::numpy, double>(data, {m_sobol.dim()}, owner);
    } catch (const std::runtime_error&) {
      throw nb::stop_iteration();
    }
  }

  SobolGenerator& iter() { return *this; }

private:
  Sobol m_sobol;
};

nb::tuple integerise(const np_array<double>& npseed) {
  const NDArray<double> seed = asNDArray<double>(npseed); // shallow copy
  Integeriser integeriser(seed);

  nb::dict stats;
  stats["conv"] = integeriser.conv(), stats["rmse"] = integeriser.rmse();
  return nb::make_tuple(fromNDArray<int64_t>(integeriser.result()), stats);
}

nb::tuple ipf(const np_array<double>& seed, const nb::iterable& index_iter, const nb::iterable& marginal_iter) {
  std::vector<std::vector<int64_t>> indices = collect_indices(index_iter);
  std::vector<NDArray<double>> marginals = collect_marginals<double>(marginal_iter);
  if (indices.size() != marginals.size())
    throw nb::value_error("index and marginals lists differ in size");

  IPF<double> ipf(indices, marginals);
  const NDArray<double>& result = ipf.solve(asNDArray<double>(seed));

  nb::dict stats;
  stats["conv"] = ipf.conv();
  stats["pop"] = ipf.population();
  stats["iterations"] = ipf.iters();
  stats["maxError"] = ipf.maxError();
  return nb::make_tuple(fromNDArray<double>(result), stats);
}

nb::tuple qis(const nb::iterable& index_iter, const nb::iterable& marginal_iter, int64_t skips) {
  std::vector<std::vector<int64_t>> indices = collect_indices(index_iter);
  std::vector<NDArray<int64_t>> marginals = collect_marginals<int64_t>(marginal_iter);
  if (indices.size() != marginals.size())
    throw nb::value_error("index and marginals lists differ in size");

  QIS qis(indices, marginals, skips);
  const NDArray<int64_t>& result = qis.solve();
  const NDArray<double>& expect = qis.expectation();

  nb::dict stats;
  stats["expectation"] = fromNDArray<double>(expect);
  stats["conv"] = qis.conv();
  stats["pop"] = qis.population();
  stats["chiSq"] = qis.chiSq(), stats["pValue"] = qis.pValue();
  stats["degeneracy"] = qis.degeneracy();
  return nb::make_tuple(fromNDArray<int64_t>(result), stats);
}

nb::tuple qisi(const np_array<double> seed, const nb::iterable& index_iter, const nb::iterable& marginal_iter,
               int64_t skips) {
  std::vector<std::vector<int64_t>> indices = collect_indices(index_iter);
  std::vector<NDArray<int64_t>> marginals = collect_marginals<int64_t>(marginal_iter);
  if (indices.size() != marginals.size())
    throw nb::value_error("index and marginals lists differ in size");

  QISI qisi(indices, marginals, skips);
  const NDArray<int64_t>& result = qisi.solve(asNDArray<double>(seed));

  nb::dict stats;
  stats["expectation"] = fromNDArray<double>(qisi.expectation());
  stats["conv"] = qisi.conv();
  stats["pop"] = qisi.population();
  stats["chiSq"] = qisi.chiSq(), stats["pValue"] = qisi.pValue();
  stats["degeneracy"] = qisi.degeneracy();

  return nb::make_tuple(fromNDArray<int64_t>(result), stats);
}

nb::dict unittest() {
  const unittest::Logger& log = unittest::run();

  nb::dict result;
  result["nTests"] = log.testsRun;
  result["nFails"] = log.testsFailed;
  result["errors"] = log.errors;

  return result;
}

} // namespace hl

NB_MODULE(humanleague_ext, m) {

#include "docstr.inl"

  m.doc() = module_docstr;

  m.def("flatten", hl::flatten, flatten_docstr, "pop"_a)
      .def("integerise", hl::integerise1d, integerise1d_docstr, "frac"_a, "pop"_a)
      .def("integerise", hl::integerise, integerise_docstr, "pop"_a)
      .def("ipf", hl::ipf, ipf_docstr, "seed"_a, "indices"_a, "marginals"_a)
      .def("qis", hl::qis, qis_docstr, "indices"_a, "marginals"_a, "skips"_a)
      .def(
          "qis",
          [](const nb::iterable& indices, const nb::iterable& marginals) { return hl::qis(indices, marginals, 0); },
          qis2_docstr, "indices"_a, "marginals"_a)
      .def("qisi", hl::qisi, qisi_docstr, "seed"_a, "indices"_a, "marginals"_a, "skips"_a)
      .def(
          "qisi",
          [](const np_array<double>& seed, const nb::iterable& indices, const nb::iterable& marginals) {
            return hl::qisi(seed, indices, marginals, 0);
          },
          qisi2_docstr, "seed"_a, "indices"_a, "marginals"_a)
      .def("_unittest", hl::unittest, unittest_docstr);

  nb::class_<hl::SobolGenerator>(m, "SobolSequence")
      .def(nb::init<size_t, size_t>(), SobolSequence_init2_docstr, "dim"_a, "skips"_a)
      .def(nb::init<size_t>(), SobolSequence_init1_docstr, "dim"_a)
      .def("__iter__", &hl::SobolGenerator::iter, "__iter__ dunder")
      .def("__next__", &hl::SobolGenerator::next, "__next__ dunder");
}

#endif
