#pragma once

#include "NDArray.h"
#include "NDArrayUtils.h"

#include <algorithm>
#include <vector>
//#include <array> sadly cant use this (easily) as length is defined in type

#include <cmath>


template<size_t D>
class IPF 
{
public:

  static const size_t Dim = D;

  IPF(const NDArray<D, double>& seed, const std::vector<std::vector<double>>& marginals)
    : m_result(seed.sizes()), m_marginals(marginals), m_errors(D), m_conv(false)
  {
    // TODO checks on marginals, dimensions etc
    m_population = std::accumulate(m_marginals[0].begin(), m_marginals[0].end(), 0);

    //print(seed.rawData(), seed.storageSize(), m_marginals[1].size());
    std::copy(seed.rawData(), seed.rawData() + seed.storageSize(), const_cast<double*>(m_result.rawData()));
    for (size_t d = 0; d < Dim; ++d)
    {
      m_errors[d].resize(m_marginals[d].size());
      //print(m_marginals[d]);
    }
    //print(m_result.rawData(), m_result.storageSize(), m_marginals[1].size());
    
    for (m_iters = 0; !m_conv && m_iters < s_MAXITER; ++m_iters) 
    {
      rScale<Dim>(m_result, m_marginals);
      // inefficient copying?
      std::vector<std::vector<double>> diffs(Dim);

      rDiff<Dim>(diffs, m_result, m_marginals);

      m_conv = computeErrors(diffs);
    }
  }
    
  IPF(const IPF&) = delete;
  IPF(IPF&&) = delete;
  
  IPF& operator=(const IPF&) = delete;
  IPF& operator=(IPF&&) = delete;
  
  virtual ~IPF() { }

  size_t population() const 
  {
    return m_population;
  }

  const NDArray<Dim, double>& result() const
  {
    return m_result;
  }

  const std::vector<std::vector<double>> errors() const 
  {
    return m_errors;
  }

  bool conv() const 
  {
    return m_conv;
  }

  size_t iters() const 
  {
    return m_iters;
  }
  
private:

  template<size_t I>
  static void rScale(NDArray<D, double>& result, const std::vector<std::vector<double>>& marginals)
  {
    const size_t Direction = I - 1;
    const std::vector<double>& r = reduce<D, double, Direction>(result);
    for (size_t p = 0; p < marginals[Direction].size(); ++p)
    {
      for (Index<Dim, Direction> index(result.sizes(), p); !index.end(); ++index) 
      {
        result[index] *= marginals[Direction][index[Direction]] / r[p]; 
      }
    }
    rScale<I-1>(result, marginals);
  }

  template<size_t I>
  static void rDiff(std::vector<std::vector<double>>& diffs, const NDArray<D, double>& result, const std::vector<std::vector<double>>& marginals)
  {
    const size_t Direction = I - 1;
    diffs[Direction] = diff(reduce<Dim, double, Direction>(result), marginals[Direction]);
    rDiff<I-1>(diffs, result, marginals);
  }

  // this is close to repeating the above 
  bool computeErrors(std::vector<std::vector<double>>& diffs)
  {
    //calcResiduals<Dim>(diffs);
    double maxError = -std::numeric_limits<double>::max();
    for (size_t d = 0; d < Dim; ++d)
    {
      for (size_t i = 0; i < diffs[d].size(); ++i)
      {
        double e = std::fabs(diffs[d][i]);
        m_errors[0][i] = e;
        maxError = std::max(maxError, e);
      }
    }
    return maxError < m_tol;
  }
  
  static const size_t s_MAXITER = 10;

  NDArray<Dim, double> m_result;
  std::vector<std::vector<double>> m_marginals;
  std::vector<std::vector<double>> m_errors;
  size_t m_population;
  size_t m_iters;
  bool m_conv;
  const double m_tol = 1e-8;
};


// Specialisation to terminate the recursion for each instantiation 
# define SPECIALISE_RSCALE(d) \
template<> \
template<> \
void IPF<d>::rScale<0>(NDArray<d, double>&, const std::vector<std::vector<double>>&) { }

SPECIALISE_RSCALE(2)
SPECIALISE_RSCALE(3)
SPECIALISE_RSCALE(4)
SPECIALISE_RSCALE(5)
SPECIALISE_RSCALE(6)
SPECIALISE_RSCALE(7)
SPECIALISE_RSCALE(8)
SPECIALISE_RSCALE(9)
SPECIALISE_RSCALE(10)
SPECIALISE_RSCALE(11)
SPECIALISE_RSCALE(12)

// Specialisation to terminate the diff for each instantiation 
# define SPECIALISE_RDIFF(d) \
template<> \
template<> \
void IPF<d>::rDiff<0>(std::vector<std::vector<double>>&, const NDArray<d, double>&, const std::vector<std::vector<double>>&) { }

SPECIALISE_RDIFF(2)
SPECIALISE_RDIFF(3)
SPECIALISE_RDIFF(4)
SPECIALISE_RDIFF(5)
SPECIALISE_RDIFF(6)
SPECIALISE_RDIFF(7)
SPECIALISE_RDIFF(8)
SPECIALISE_RDIFF(9)
SPECIALISE_RDIFF(10)
SPECIALISE_RDIFF(11)
SPECIALISE_RDIFF(12)

// disable trivial and nonsensical dimensionalities
template<> class IPF<0>;
template<> class IPF<1>;

// remove macro pollution
#undef SPECIALISE_RSCALE
#undef SPECIALISE_RDIFF
