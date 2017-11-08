
#include "QIS.h"
#include "Index.h"
#include "StatFuncs.h"

namespace {

  // TODO move somewhere appropriate (doesnt need to be member) (copy&paste from QSIPF)
template<typename T>
// TODO take raw memory to avoid copying to vector
size_t pick(const std::vector<T>& dist, double r)
{
  // sum of dist should be 1, but we relax this
  // r is in (0,1) so scale up r by sum of dist
  r *= std::accumulate(dist.begin(), dist.end(), 0.0);
  T runningSum = 0.0;
  for (size_t i = 0; i < dist.size(); ++i)
  {
    runningSum += dist[i];
    if (r < runningSum)
      return i;
  }
  throw std::runtime_error("pick failed");
}

template<typename T>
void getIndex(const NDArray<T>& p, const std::vector<uint32_t>& r, Index& index)
{
  static const double scale = 0.5 / (1u<<31);

  size_t dim = p.dim();

  if (dim > 2)
  {
    // reduce dim D-1
    const std::vector<T>& m = reduce<T>(p, dim - 1);
    // pick an index
    index[dim-1] = pick(m, r[dim-1] * scale);

    // take slice of Dim D-1 at index
    const NDArray<T>& sliced = slice<T>(p, {dim-1, index[dim-1]});

    // recurse
    getIndex(sliced, r, index);
  }
  else if (dim == 2)
  {
    // reduce dim 1 (now 0)
    const std::vector<T>& r1 = reduce<T>(p, 1);
    // pick an index
    index[1] = pick(r1, r[1] * scale);

    // slice dim 2 (now 0)
    const NDArray<T>& sliced = slice<T>(p, {1, index[1]});
    assert(sliced.dim() == 1);
    std::vector<T> r0(sliced.rawData(), sliced.rawData() + sliced.storageSize());
    // no reduction required
    // pick an index
    index[0] = pick(r0, r[0] * scale);
  }
  else
  {
    const std::vector<T> r0(p.rawData(), p.rawData() + p.storageSize());
    index[0] = pick(r0, r[0] * scale);
  }
}

}

QIS::QIS(const index_list_t& indices, marginal_list_t& marginals, int64_t skips)
: Microsynthesis(indices, marginals), m_sobolSeq(m_dim), m_conv(false)
{
  m_sobolSeq.skip(skips);
  //m_sobolSeq = Sobol(m_dim);
  m_stateProbs.resize(m_array.sizes());
  // compute initial state probabilities and keep a copy
  updateStateProbs();
  NDArray<double>::copy(m_stateProbs, m_expectedStateOccupancy);
  // scale up
  for (Index index(m_expectedStateOccupancy.sizes()); !index.end(); ++index)
    m_expectedStateOccupancy[index] *= m_population;
}

// // control state of Sobol via arg?
// const NDArray<int64_t>& QIS::solve2()
// {
//   m_conv = true;
//   Index main_index(m_array.sizes());
//   const std::vector<MappedIndex>& mappedIndices = makeMarginalMappings(main_index);
//   m_array.assign(0ll);

//   // this is massively inefficent, far better to sample directly from each marginal
//   Sobol sobol_seq(m_dim);
//   for (int64_t i = 0; i < m_population; ++i)
//   {
//     // map sobol to a point in state space, store in index
//     const std::vector<uint32_t>& seq = sobol_seq.buf();
//     // ...
//     getIndex(m_stateProbs, seq, main_index);

//     // increment population
//     ++m_array[main_index];

//     // decrement marginals, checking none have gone -ve
//     for (size_t j = 0; j < mappedIndices.size(); ++j)
//     {
//       --m_marginals[j][mappedIndices[j]];
//       if (m_marginals[j][mappedIndices[j]] < 0)
//         m_conv = false;
//     }
//     // recalculate state probabilities
//     updateStateProbs();
//   }
//   m_chiSq = ::chiSq(m_array, m_expectedStateOccupancy);

//   m_pValue = ::pValue(dof(m_array.sizes()), m_chiSq).first;

//   m_degeneracy = ::degeneracy(m_array);

//   return m_array;
// }

const NDArray<int64_t>& QIS::solve(bool reset)
{
  if (reset)
  {
    m_sobolSeq.reset();
  }

  m_conv = true;
  // loop over population
  m_array.assign(0ll);

  for (int64_t i = 0; i < m_population; ++i)
  {
    // init main_index with unset values
    std::vector<int64_t> main_index(m_array.sizes().size(), -1/*Index::Unfixed*/);
    // take values from Sobol
    const std::vector<uint32_t>& seq = m_sobolSeq.buf();

    // loop over marginals
    for (size_t k = 0; k < m_marginals.size(); ++k)
    {
      Index index(m_marginals[k].sizes());
      // take relevant part(s) of Sobol
      std::vector<uint32_t> r;
      r.reserve(m_marginals[k].dim());
      for (auto q : m_indices[k])
        r.push_back(seq[q]);

      // TODO need a way of dealing with already-picked indices
      getIndex(m_marginals[k], r, index);
      // insert indices into main_index (where unset!)
      for (size_t j = 0; j < m_indices[k].size(); ++j)
      {
        // TODO try to find a case whexe index gets changed (or prove its not possible)
#ifndef NDEBUG
        if (main_index[m_indices[k][j]] != -1/*Index::Unfixed*/ && main_index[m_indices[k][j]] != index[j])
        {
          std::cout << std::to_string(k) << ": changing " << std::to_string(main_index[m_indices[k][j]]) << " to " << std::to_string(index[j]) << std::endl;
        }
#endif
        main_index[m_indices[k][j]] = index[j];
      }
      // create index for that marginal by sampling sobol values

      // decrement marginal
      --m_marginals[k][index];
      if (m_marginals[k][index] < 0)
        m_conv = false;
    }
    // increment pop
    ++m_array[main_index];
  }
  m_chiSq = ::chiSq(m_array, m_expectedStateOccupancy);

  m_pValue = ::pValue(dof(m_array.sizes()), m_chiSq).first;

  m_degeneracy = ::degeneracy(m_array);

  return m_array;
}

// control state of Sobol via arg?
// better solution? construct set of 1-d marginals and sample from these
const NDArray<int64_t>& QIS::solve3(bool reset)
{
  if (reset)
  {
    m_sobolSeq.reset();
  }

  m_conv = true;
  // loop over population
  m_array.assign(0ll);

  // compile list of marginals present in each dimension
  //std::vector<std::vector<size_t>> dim_lookup(m_dim);
  // // loop over dimensions
  // for (size_t k = 0; k < m_dim; ++k)
  // {
  //   // loop over marginal indices, finding ones relevant to this dimension
  //   for (size_t m = 0; m < m_indices.size(); ++m)
  //   {
  //     auto it = std::find(m_indices[m].begin(), m_indices[m].end(), k);
  //     if (it != m_indices[m].end())
  //     {
  //       std::cout << "marginal " << m << " has dim " << k << std::endl;
  //       dim_lookup[k].push_back(m);
  //     }
  //   }
  // }

  Index main_index(m_array.sizes());
  
  const std::vector<int64_t>& main_index_raw = main_index;
  std::vector<std::vector<int64_t>> mapped_indices_raw(m_marginals.size());

  std::cout << "Dim Mrg,Dim" << std::endl;
  for (size_t d = 0; d < m_dim_lookup.size(); ++d)
  {
    std::cout << d << ":";
    for (size_t i = 0; i < m_dim_lookup[d].size(); ++i)
    {
      std::cout << m_dim_lookup[d][i].first << "," << m_dim_lookup[d][i].second << " ";
      mapped_indices_raw[m_dim_lookup[d][i].first].push_back(d);
    }
    std::cout << std::endl;
  }

  //print(main_index_raw);

  std::vector<MappedIndex> mapped_indices;
  mapped_indices.reserve(m_marginals.size());
  for (size_t i = 0; i < m_marginals.size(); ++i)
  {
    mapped_indices.push_back(MappedIndex(main_index, mapped_indices_raw[i]));
  }

  // mark main index as unassigned
  for (size_t d = 0; d < m_dim; ++d)
  {
    main_index[d] = -1;
  }

  // check 
  for (size_t m = 0; m < mapped_indices.size(); ++m)
  {
    const std::vector<int64_t*>& x = mapped_indices[m];
    for (size_t i = 0; i < x.size(); ++i)
    {
      std::cout << *x[i];
    }
    std::cout << std::endl;
  }


  for (int64_t i = 0; i < m_population; ++i)
  {
    // take values from Sobol
    const std::vector<uint32_t>& seq = m_sobolSeq.buf();

    // loop over marginals
    for (size_t m = 0; m < mapped_indices.size(); ++m)
    {
      sample(seq, m_marginals[m], mapped_indices[m]);

    }

  }

  // for (int64_t i = 0; i < m_population; ++i)
  // {
  //   // init main_index with unset values
  //   std::vector<int64_t> main_index(m_array.sizes().size(), -1/*Index::Unfixed*/);
  //   // take values from Sobol
  //   const std::vector<uint32_t>& seq = m_sobolSeq.buf();

    // // loop over dimensions
    // for (size_t k = 0; k < m_dim; ++k)
    // {
    //   for (size_t m = 0; m < dim_lookup[k].size(); ++m)
    //   {
    //
    //   }
    // }

//     // loop over marginals
//     for (size_t k = 0; k < m_marginals.size(); ++k)
//     {
//       Index index(m_marginals[k].sizes());
//       // take relevant part(s) of Sobol
//       std::vector<uint32_t> r;
//       r.reserve(m_marginals[k].dim());
//       for (auto q : m_indices[k])
//         r.push_back(seq[q]);

//       // TODO need a way of dealing with already-picked indices
//       getIndex(m_marginals[k], r, index);
//       // insert indices into main_index (where unset!)
//       for (size_t j = 0; j < m_indices[k].size(); ++j)
//       {
//         // TODO try to find a case whexe index gets changed (or prove its not possible)
// //#ifndef NDEBUG
//         if (main_index[m_indices[k][j]] != -1/*Index::Unfixed*/ && main_index[m_indices[k][j]] != index[j])
//         {
//           std::cout << std::to_string(k) << ": QIS changing " << std::to_string(main_index[m_indices[k][j]]) << " to " << std::to_string(index[j]) << std::endl;
//         }
//         else
//         {
// //#endif
//           main_index[m_indices[k][j]] = index[j];
// //#ifndef NDEBUG
//         }
// //#endif
//       }
//       // create index for that marginal by sampling sobol values

    //   // decrement marginal
    //   --m_marginals[k][index];
    //   if (m_marginals[k][index] < 0)
    //     m_conv = false;
    // }
    // increment pop
  //  ++m_array[main_index];
  //}
  m_chiSq = ::chiSq(m_array, m_expectedStateOccupancy);

  m_pValue = ::pValue(dof(m_array.sizes()), m_chiSq).first;

  m_degeneracy = ::degeneracy(m_array);

  return m_array;
}

void QIS::sample(const std::vector<uint32_t>& seq, const NDArray<int64_t>& marginal, MappedIndex& index)
{
//   // TODO if no unset values, do nothing

//   std::vector<int64_t> unsampled;
//   for (size_t d = 0; d < marginal.dim(); ++d)
//   {
//     if (index[d] == -1)
//       unsampled.push_back(d);
//   }

//   // create reduced array in unsampled dims
//   NDArray<int64_t>& r = std::move(reduce(marginal, unsampled));

}

// Expected state occupancy
const NDArray<double>& QIS::expectation()
{
  return m_expectedStateOccupancy;
}

//
void QIS::updateStateProbs()
{
  Index index_main(m_array.sizes());

  std::vector<MappedIndex> mappings = makeMarginalMappings(index_main);

  m_stateProbs.assign(1.0);
  for (; !index_main.end(); ++index_main)
  {
    for (size_t k = 0; k < m_marginals.size(); ++k)
    {
      m_stateProbs[index_main] *= m_marginals[k][mappings[k]];
    }
  }

  // TODO rescaling is unnecessary
  double scale = 1.0 / sum(m_stateProbs);
  for (Index index(m_array.sizes()); !index.end(); ++index)
  {
    m_stateProbs[index] *= scale;
  }
}

double QIS::chiSq() const
{
  return m_chiSq;
}

double QIS::pValue() const
{
  return m_pValue;
}

double QIS::degeneracy() const
{
  return m_degeneracy;
}

bool QIS::conv() const
{
  return m_conv;
}

