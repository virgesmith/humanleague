

#include "IPF.h"
#include "NDArrayUtils.h"

#include <algorithm>

#include <cmath>

namespace {

void rScale(NDArray<double>& result, const std::vector<std::vector<double>>& marginals)
{
  for (size_t d = 0; d < result.dim(); ++d)
  {
    const std::vector<double>& r = reduce<double>(result, d);
    for (size_t p = 0; p < marginals[d].size(); ++p)
    {
      for (Index index(result.sizes(), { d, p }); !index.end(); ++index)
      {
        const std::vector<int64_t>& ref = index;
        // avoid division by zero (assume 0/0 -> 0)
        if (r[p] == 0.0 && marginals[d][ref[d]] != 0.0)
          throw std::runtime_error("div0 in rScale with m>0");
        if (r[p] != 0.0)
          result[index] *= marginals[d][ref[d]] / r[p];
        else
          result[index] = 0.0;
      }
    }
  }
}

void rDiff(std::vector<std::vector<double>>& diffs, const NDArray<double>& result, const std::vector<std::vector<double>>& marginals)
{
  int64_t n = result.dim();
  for (int64_t d = 0; d < n; ++d)
    diffs[d] = diff(reduce<double>(result, d), marginals[d]);
}

}

// construct from fractional marginals
IPF::IPF(const NDArray<double>& seed, const std::vector<std::vector<double>>& marginals)
  : m_result(seed.sizes()), m_marginals(marginals), m_errors(seed.dim()), m_conv(false)
{
  if (m_marginals.size() != seed.dim())
    throw std::runtime_error("no. of marginals doesnt match dimensionalty");
  solve(seed);
}

// construct from integer marginals
IPF::IPF(const NDArray<double>& seed, const std::vector<std::vector<int64_t>>& marginals)
: m_result(seed.sizes()), m_marginals(marginals.size()), m_errors(marginals.size()), m_conv(false)
{
  if (marginals.size() != seed.dim())
    throw std::runtime_error("no. of marginals doesnt match dimensionalty");
  for (size_t d = 0; d < seed.dim(); ++d)
  {
    m_marginals[d].reserve(marginals[d].size());
    std::copy(marginals[d].begin(), marginals[d].end(), std::back_inserter(m_marginals[d]));
  }
  solve(seed);
}

void IPF::solve(const NDArray<double>& seed)
{
  // reset convergence flag
  m_conv = false;
  m_population = std::accumulate(m_marginals[0].begin(), m_marginals[0].end(), 0.0);
  // checks on marginals, dimensions etc
  for (size_t i = 0; i < seed.dim(); ++i)
  {
    if ((int64_t)m_marginals[i].size() != m_result.sizes()[i])
      throw std::runtime_error("marginal doesnt have correct length");

    double mpop = std::accumulate(m_marginals[i].begin(), m_marginals[i].end(), 0.0);
    if (mpop != m_population)
      throw std::runtime_error("marginal doesnt have correct population");
  }

  //print(seed.rawData(), seed.storageSize(), m_marginals[1].size());
  std::copy(seed.rawData(), seed.rawData() + seed.storageSize(), const_cast<double*>(m_result.rawData()));
  for (size_t d = 0; d < m_result.dim(); ++d)
  {
    m_errors[d].resize(m_marginals[d].size());
    //print(m_marginals[d]);
  }
  //print(m_result.rawData(), m_result.storageSize(), m_marginals[1].size());

  std::vector<std::vector<double>> diffs(m_result.dim());
  for (m_iters = 0; !m_conv && m_iters < s_MAXITER; ++m_iters)
  {
    rScale(m_result, m_marginals);
    // inefficient copying?

    rDiff(diffs, m_result, m_marginals);

    m_conv = computeErrors(diffs);
  }
}

size_t IPF::population() const
{
  return m_population;
}

const NDArray<double>& IPF::result() const
{
  return m_result;
}

const std::vector<std::vector<double>> IPF::errors() const
{
  return m_errors;
}

double IPF::maxError() const
{
  return m_maxError;
}

bool IPF::conv() const
{
  return m_conv;
}

size_t IPF::iters() const
{
  return m_iters;
}

bool IPF::computeErrors(std::vector<std::vector<double>>& diffs)
{
  m_maxError = -std::numeric_limits<double>::max();
  for (size_t d = 0; d < m_result.dim(); ++d)
  {
    for (size_t i = 0; i < diffs[d].size(); ++i)
    {
      double e = std::fabs(diffs[d][i]);
      m_errors[d][i] = e;
      m_maxError = std::max(m_maxError, e);
    }
  }
  return m_maxError < m_tol;
}

namespace wip {

// TODO perhaps seed should be an arg to solve instead of being passed in here
IPF::IPF(const index_list_t& indices, marginal_list_t& marginals)
  : Microsynthesis(indices, marginals)
{

}

NDArray<double>& IPF::solve(const NDArray<double>& seed)
{
  // check seed dims match those computed by base
  assert(seed.sizes() == m_array.sizes());

  Index index_main(m_array.sizes());

  std::vector<MappedIndex> mappings;
  mappings.reserve(m_marginals.size());
  for (size_t k = 0; k < m_marginals.size(); ++k)
  {
    mappings.push_back(MappedIndex(index_main, m_indices[k]));
  }

  m_array.assign(1.0);
  std::copy(seed.rawData(), seed.rawData() + seed.storageSize(), const_cast<double*>(m_array.rawData()));

  marginal_list_t diffs(m_marginals.size());
  m_errors.resize(m_marginals.size());

  for (size_t k = 0; k < diffs.size(); ++k)
  {
    diffs[k].resize(m_marginals[k].sizes());
    m_errors[k].resize(m_marginals[k].sizes());
  }

  m_conv = false;
  for (m_iters = 0; !m_conv && m_iters < s_MAXITER; ++m_iters)
  {
    rScale(/*m_array, m_marginals*/);
    rDiff(diffs);

    m_conv = computeErrors(diffs);
  }

  return m_array;
}

const std::vector<NDArray<double>>& IPF::errors() const
{
  return m_errors;
}

double IPF::maxError() const
{
  return m_maxError;
}

bool IPF::conv() const
{
  return m_conv;
}

size_t IPF::iters() const
{
  return m_iters;
}


bool IPF::computeErrors(std::vector<NDArray<double>>& diffs)
{
  m_maxError = -std::numeric_limits<double>::max();

  // // create mapped indices
  // const std::vector<MappedIndex>& mapped = makeMappings(main_index);
  for (size_t k = 0; k < diffs.size(); ++k)
  {
    for (Index index(diffs[k].sizes()); !index.end(); ++index)
    {
      double e = std::fabs(diffs[k][index]);
      m_errors[k][index] = e;
      m_maxError = std::max(m_maxError, e);
    }
  }

  return m_maxError < m_tol;
}

void IPF::rScale()
{
  // for (size_t d = 0; d < result.dim(); ++d)
  // {
  //   const std::vector<double>& r = reduce<double>(result, d);
  //   for (size_t p = 0; p < marginals[d].size(); ++p)
  //   {
  //     for (Index index(result.sizes(), { d, p }); !index.end(); ++index)
  //     {
  //       const std::vector<int64_t>& ref = index;
  //       // avoid division by zero (assume 0/0 -> 0)
  //       if (r[p] == 0.0 && marginals[d][ref[d]] != 0.0)
  //         throw std::runtime_error("div0 in rScale with m>0");
  //       if (r[p] != 0.0)
  //         result[index] *= marginals[d][ref[d]] / r[p];
  //       else
  //         result[index] = 0.0;
  //     }
  //   }
  // }
  for (size_t k = 0; k < m_indices.size(); ++k)
  {
    const NDArray<double>& r = reduce<double>(m_array, m_indices[k]);
    // std::cout << k << ":";
    // print(r.rawData(), r.storageSize());


    Index main_index(m_array.sizes());
    //std::cout << m_array.sizes()[m_indices[1-k][0]] << std::endl;
    for (MappedIndex oindex(main_index, invert(m_array.dim(), m_indices[k])); !oindex.end(); ++oindex)
    {
      for (MappedIndex index(main_index, m_indices[k]); !index.end(); ++index)
      {
        //print((std::vector<int64_t>)main_index);
        if (r[index] == 0.0 && m_marginals[k][index] != 0.0)
          throw std::runtime_error("div0 in rScale with m>0");
        if (r[index] != 0.0)
          m_array[main_index] *= m_marginals[k][index] / r[index];
        else
          m_array[main_index] = 0.0;
      }
    }
    // reset the main index
    //main_index.reset();
  }
}

void IPF::rDiff(std::vector<NDArray<double>>& diffs)
{
  int64_t n = m_indices.size();
  for (int64_t k = 0; k < n; ++k)
    diff(reduce<double>(m_array, m_indices[k]), m_marginals[k], diffs[k]);
}


} // wip
