// Microsynthesis.h
// Basic functionality to hold and iterate over mulitdimensional marginals

#pragma once

#include "NDArray.h"
#include "NDArrayUtils.h"
#include "Index.h"
#include "Log.h"

#include <vector>
#include <map>

#include <string>
#include <stdexcept>

// T = population, M = marginals (defaults to same type)
template<typename T, typename M = T>
class Microsynthesis
{
public:
  typedef std::vector<int64_t> index_t;
  typedef std::vector<index_t> index_list_t;
  typedef NDArray<M> marginal_t;
  typedef std::vector<marginal_t> marginal_list_t;

  typedef std::vector<std::pair<int64_t, int64_t>> marginal_indices_t;
  typedef std::vector<marginal_indices_t> marginal_indices_list_t;

  Microsynthesis(const index_list_t& indices, marginal_list_t& marginals):
    m_indices(indices), m_marginals(marginals)
  {
    // i and m should be same size and >2
    if (m_indices.size() != m_marginals.size() || m_indices.size() < 2)
      throw std::runtime_error("index list size %% too small or differs from marginal size %%"_s % m_indices.size() % m_marginals.size());

    // count all unique values in i...
    std::map<int64_t, int64_t> dim_sizes;
    for (size_t k = 0; k < m_indices.size(); ++k)
    {
      if (m_indices[k].size() != m_marginals[k].dim())
        throw std::runtime_error("index/marginal dimension mismatch %% vs %%"_s % m_indices[k].size() % m_marginals[k].dim());
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
          throw std::runtime_error("mismatch at index %%: dimension %% size %% redefined to %%"_s % k % dim % posit->second % size);
      }
    }

    if (dim_sizes.size() < 2)
      throw std::runtime_error("problem needs to have more than 1 dimension!");

    // validate marginals
    for (size_t k = 0; k < m_marginals.size(); ++k)
    {
      if (min(m_marginals[k]) < 0)
        throw std::runtime_error("negative value in marginal %%: %%"_s % k % min(m_marginals[k]));
    }

    m_dim = dim_sizes.size();
    
    // check all dims defined
    //std::vector<int64_t> sizes;
    m_sizes.reserve(m_indices.size());

    // we should expect that the dim_sizes map contains keys for all values in [ 0, dim_sizes.size() ). if not throw
    for (size_t k = 0; k < dim_sizes.size(); ++k)
    {
      auto it = dim_sizes.find(k);
      if (it == dim_sizes.end())
        throw std::runtime_error("dimension %% size not defined"_s % k);
      m_sizes.push_back(it->second);
    }

    createMappings(m_sizes, dim_sizes);

    m_array.resize(m_sizes);

#ifdef VERBOSE
    // print summary data
    std::cout << "Dim Size" << std::endl;
    for (size_t d = 0; d < m_sizes.size(); ++d)
    {
        std::cout << d << ": " << m_sizes[d] << std::endl;
    }

    std::cout << "Mrg Dims" << std::endl;
    for (size_t k = 0; k < m_indices.size(); ++k)
    {
        std::cout << k << ": ";
        print(m_indices[k]);
    }

    std::cout << "Dim Mrg,Dim" << std::endl;
    for (size_t d = 0; d < m_dim_lookup.size(); ++d)
    {
        std::cout << d << ":";
        for (size_t i = 0; i < m_dim_lookup[d].size(); ++i)
        {
        std::cout << m_dim_lookup[d][i].first << "," << m_dim_lookup[d][i].second << " ";
        }
        std::cout << std::endl;
    }
#endif
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

  // Diffs always represented in floating point
  void rDiff(std::vector<NDArray<double>>& diffs)
  {
    int64_t n = m_indices.size();
    for (int64_t k = 0; k < n; ++k)
      diff(reduce<double>(m_array, m_indices[k]), m_marginals[k], diffs[k]);
  }

protected:
  
  void rScale()
  {
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
#ifndef NDEBUG
          if (r[index] == 0.0 && m_marginals[k][index] != 0.0)
            throw std::runtime_error("div0 in rScale with m>0");
          if (r[index] != 0.0)
            m_array[main_index] *= m_marginals[k][index] / r[index];
          else
            m_array[main_index] = 0.0;
#else
          if (r[index] != 0.0)
            m_array[main_index] *= m_marginals[k][index] / r[index];
          else
            m_array[main_index] = 0.0;
#endif  
        }
      }
      // reset the main index
      //main_index.reset();
    }
  }
  
  void createMappings(const std::vector<int64_t> sizes, const std::map<int64_t, int64_t>& dim_sizes)
  {
    // create mapping from dimension to marginal(s)
    m_dim_lookup.resize(m_dim);

    for (size_t k = 0; k < m_indices.size(); ++k)
      for (size_t i = 0; i < m_indices[k].size(); ++i)
        m_dim_lookup[m_indices[k][i]].push_back(std::make_pair(k,i));

    // more validation

    // check marginal sums all the same
    m_population = static_cast<int64_t>(sum(m_marginals[0]));
    for (size_t i = 1; i < m_marginals.size(); ++i)
    {
      if (static_cast<int64_t>(sum(m_marginals[i])) != m_population)
        throw std::runtime_error("marginal sum mismatch at index %%: %% vs %%"_s % i % sum(m_marginals[i]) % m_population);
    }

    // check that for each dimension included in more than one marginal, the partial sums in that dimension are equal
    for (size_t d = 0; d < m_dim; ++d)
    {
      // loop over the relevant marginals
      const marginal_indices_t& mi = m_dim_lookup[d];
      if (mi.size() < 2)
        continue;
      //                                marginal index            marginal dimension
      const std::vector<M>& ms = reduce(m_marginals[mi[0].first], mi[0].second);
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
  // lists marginals and dims of marginals per overall dimension
  marginal_indices_list_t m_dim_lookup;
  NDArray<T> m_array;
};
