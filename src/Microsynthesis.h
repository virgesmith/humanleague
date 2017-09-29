// Microsynthesis.h
// Basic functionality to hold and iterate over mulitdimensional marginals

#pragma once

#include "NDArray.h"
#include "NDArrayUtils.h"
#include "Index.h"

#include <vector>
#include <map>

#include <iostream>

template<typename T>
class Microsynthesis
{
public:
  typedef std::vector<int64_t> index_t;
  typedef std::vector<index_t> index_list_t;
  typedef NDArray<T> marginal_t;
  typedef std::vector<marginal_t> marginal_list_t;

  typedef std::vector<std::pair<int64_t, int64_t>> marginal_indices_t;
  typedef std::vector<marginal_indices_t> marginal_indices_list_t;

  Microsynthesis(const index_list_t& indices, marginal_list_t& marginals):
    m_indices(indices), m_marginals(marginals)
  {
    // i and m should be same size and >2
    if (m_indices.size() != m_marginals.size() || m_indices.size() < 2)
      throw std::runtime_error("index and marginal lists differ in size or too small");

    // count all unique values in i...
    std::map<int64_t, int64_t> dim_sizes;
    for (size_t k = 0; k < m_indices.size(); ++k)
    {
      if (m_indices[k].size() != m_marginals[k].dim())
        throw std::runtime_error("index/marginal dimension mismatch " + std::to_string(m_indices[k].size()) + " vs " + std::to_string(m_marginals[k].dim()));
      //std::cout << "index " << k << std::endl;
      for (size_t j = 0; j < m_indices[k].size(); ++j)
      {
        int64_t dim = m_indices[k][j];
        int64_t size = m_marginals[k].size(j);
        // check if already entered that size is same
        auto posit = dim_sizes.find(dim);
        if (posit == dim_sizes.end())
          dim_sizes.insert(std::make_pair(dim, size));
        else if (posit->second != size)
          throw std::runtime_error("mismatch at index " + std::to_string(k) +
            ": dimension " + std::to_string(dim) + " size " + std::to_string(posit->second) + " redefined to " + std::to_string(size));
        //std::cout << "  " << dim << ":" << size << std::endl;
      }
    }

    // check all dims defined
    //std::vector<int64_t> sizes;
    m_sizes.reserve(m_indices.size());

    // we should expect that the dim_sizes map contains keys for all values in [ 0, dim_sizes.size() ). if not throw
    for (size_t k = 0; k < dim_sizes.size(); ++k)
    {
      auto it = dim_sizes.find(k);
      if (it == dim_sizes.end())
        throw std::runtime_error("dimension " + std::to_string(k) + " size not defined");
      m_sizes.push_back(it->second);
    }

    createMappings(m_sizes, dim_sizes);

    m_array.resize(m_sizes);

    // // print summary data
    // std::cout << "Dim Size" << std::endl;
    // for (size_t d = 0; d < m_sizes.size(); ++d)
    // {
    //     std::cout << d << ": " << m_sizes[d] << std::endl;
    // }

    // std::cout << "Mrg Dims" << std::endl;
    // for (size_t k = 0; k < m_indices.size(); ++k)
    // {
    //     std::cout << k << ": ";
    //     print(m_indices[k]);
    // }

    // std::cout << "Dim Mrg,Dim" << std::endl;
    // for (size_t d = 0; d < m_dim_lookup.size(); ++d)
    // {
    //     std::cout << d << ":";
    //     for (size_t i = 0; i < m_dim_lookup[d].size(); ++i)
    //     {
    //     std::cout << m_dim_lookup[d][i].first << "," << m_dim_lookup[d][i].second << " ";
    //     }
    //     std::cout << std::endl;
    // }
  }

  Microsynthesis(const Microsynthesis&) = delete;
  Microsynthesis(Microsynthesis&&) = delete;

  Microsynthesis& operator=(const Microsynthesis&) = delete;
  Microsynthesis& operator=(Microsynthesis&&) = delete;


  virtual ~Microsynthesis() { }

  std::vector<MappedIndex> makeMarginalMappings(const Index& index_main) const
  {
    std::vector<MappedIndex> mappings;
    mappings.reserve(m_marginals.size());
    for (size_t k = 0; k < m_marginals.size(); ++k)
    {
      mappings.push_back(MappedIndex(index_main, m_indices[k]));
    }
    return mappings;
  }


  // Selects indices in [0,max) that arent in excluded
  // TODO move to a more appropriate place
  std::vector<int64_t> invert(size_t max, const std::vector<int64_t>& excluded)
  {
    //std::cout << "invert " << max << std::endl;
    //print(excluded);
    std::vector<int64_t> included;
    included.reserve(max - excluded.size());
    for (size_t i = 0; i < max; ++i)
    {
      if (std::find(excluded.begin(), excluded.end(), i) == excluded.end())
        included.push_back(i);
    }
    //print(included);
    return included;
  }

  std::vector<int64_t> sizes() const
  {
    return m_sizes;
  }

  int64_t population() const
  {
    return m_population;
  }


protected:

  void createMappings(const std::vector<int64_t> sizes, const std::map<int64_t, int64_t>& dim_sizes)
  {
    // create mapping from dimension to marginal(s)
    m_dim = dim_sizes.size();
    m_dim_lookup.resize(m_dim);

    for (size_t k = 0; k < m_indices.size(); ++k)
    for (size_t i = 0; i < m_indices[k].size(); ++i)
        m_dim_lookup[m_indices[k][i]].push_back(std::make_pair(k,i));

    // more validation

    // check marginal sums all the same
    m_population = sum(m_marginals[0]);
    for (size_t i = 1; i < m_marginals.size(); ++i)
    {
      if (sum(m_marginals[i]) != m_population)
        throw std::runtime_error("marginal sum mismatch");
    }

    // check that for each dimension included in more than one marginal, the partial sums in that dimension are equal
    for (size_t d = 0; d < m_dim; ++d)
    {
      // loop over the relevant marginals
      const marginal_indices_t& mi = m_dim_lookup[d];
      if (mi.size() < 2)
        continue;
      //                                     marginal index            marginal dimension
      const std::vector<T>& ms = reduce(m_marginals[mi[0].first], mi[0].second);
      for (size_t i = 1; i < mi.size(); ++i)
      {
        if (reduce(m_marginals[mi[i].first], mi[i].second) != ms)
        throw std::runtime_error("marginal partial sum mismatch");
      }
    }
  }

  size_t m_dim;
  std::vector<int64_t> m_sizes;
  index_list_t m_indices;
  // TODO not a ref
  marginal_list_t& m_marginals;
  int64_t m_population;
  marginal_indices_list_t m_dim_lookup;
  NDArray<T> m_array;
};
