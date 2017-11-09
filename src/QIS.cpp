
#include "QIS.h"
#include "Index.h"
#include "StatFuncs.h"

namespace {

// TODO move somewhere appropriate (doesnt need to be member) (copy&paste from QSIPF)
template<typename T>
int64_t pick(const T* dist, size_t len, double r)
{
  // sum of dist should be 1, but we relax this
  // r is in (0,1) so scale up r by sum of dist
  r *= std::accumulate(dist, dist + len, 0.0);
  T runningSum = 0.0;
  for (size_t i = 0; i < len; ++i)
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
    index[dim-1] = pick(m.data(), m.size(), r[dim-1] * scale);

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
    index[1] = pick(r1.data(), r1.size(), r[1] * scale);

    // slice dim 2 (now 0)
    const NDArray<T>& sliced = slice<T>(p, {1, index[1]});
    assert(sliced.dim() == 1);
    // no reduction required
    // pick an index
    index[0] = pick(sliced.rawData(), sliced.storageSize(), r[0] * scale);
  }
  else
  {
    index[0] = pick(p.rawData(), p.storageSize(), r[0] * scale);
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


// const NDArray<int64_t>& QIS::solve3(bool reset)
// {
//   if (reset)
//   {
//     m_sobolSeq.reset();
//   }

//   m_conv = true;
//   // loop over population
//   m_array.assign(0ll);

//   for (int64_t i = 0; i < m_population; ++i)
//   {
//     // init main_index with unset values
//     std::vector<int64_t> main_index(m_array.sizes().size(), -1/*Index::Unfixed*/);
//     // take values from Sobol
//     const std::vector<uint32_t>& seq = m_sobolSeq.buf();

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
// #ifndef NDEBUG
//         if (main_index[m_indices[k][j]] != -1/*Index::Unfixed*/ && main_index[m_indices[k][j]] != index[j])
//         {
//           std::cout << std::to_string(k) << ": changing " << std::to_string(main_index[m_indices[k][j]]) << " to " << std::to_string(index[j]) << std::endl;
//         }
// #endif
//         main_index[m_indices[k][j]] = index[j];
//       }
//       // create index for that marginal by sampling sobol values

//       // decrement marginal
//       --m_marginals[k][index];
//       if (m_marginals[k][index] < 0)
//         m_conv = false;
//     }
//     // increment pop
//     ++m_array[main_index];
//   }
//   m_chiSq = ::chiSq(m_array, m_expectedStateOccupancy);

//   m_pValue = ::pValue(dof(m_array.sizes()), m_chiSq).first;

//   m_degeneracy = ::degeneracy(m_array);

//   return m_array;
// }

// control state of Sobol via arg?
// better solution? construct set of 1-d marginals and sample from these
const NDArray<int64_t>& QIS::solve(bool reset)
{
  if (reset)
  {
    m_sobolSeq.reset();
  }

  m_conv = true;
  // loop over population
  m_array.assign(0ll);

  Index main_index(m_array.sizes());
  
  std::vector<std::vector<int64_t>> mapped_indices_raw(m_marginals.size());

  // std::cout << "Mrg, Dims" << std::endl;
  // for (size_t m = 0; m < m_indices.size(); ++m)
  // {
  //   std::cout << m << ": ";
  //   print(m_indices[m]);
  // }

  //std::cout << "Dim Mrg,Dim" << std::endl;
  for (size_t d = 0; d < m_dim_lookup.size(); ++d)
  {
    //std::cout << d << ":";
    for (size_t i = 0; i < m_dim_lookup[d].size(); ++i)
    {
      //std::cout << m_dim_lookup[d][i].first << "," << m_dim_lookup[d][i].second << " ";
      mapped_indices_raw[m_dim_lookup[d][i].first].push_back(d);
    }
    //std::cout << std::endl;
  }

  //print(main_index_raw);

  std::vector<MappedIndex> mapped_indices;
  mapped_indices.reserve(m_marginals.size());
  for (size_t i = 0; i < m_marginals.size(); ++i)
  {
    mapped_indices.push_back(MappedIndex(main_index, mapped_indices_raw[i]));
  }

  // // check 
  // for (size_t m = 0; m < mapped_indices.size(); ++m)
  // {
  //   const std::vector<int64_t*>& x = mapped_indices[m];
  //   for (size_t i = 0; i < x.size(); ++i)
  //   {
  //     std::cout << *x[i];
  //   }
  //   std::cout << std::endl;
  // }


  for (int64_t i = 0; i < m_population; ++i)
  {
    // mark main index as unassigned
    for (size_t d = 0; d < m_dim; ++d)
    {
      main_index[d] = -1;
    }

    // take values from Sobol
    const std::vector<uint32_t>& seq = m_sobolSeq.buf();

    // loop over marginals (re)sampling until main_index is populated 
    for (size_t m = 0; m < mapped_indices.size(); ++m)
    {
      sample(seq, m, mapped_indices[m]);
      //print(main_index.operator const std::vector<int64_t, std::allocator<int64_t>> &());
    }
    for (size_t m = 0; m < mapped_indices.size(); ++m)
    {
      --m_marginals[m][mapped_indices[m]];
      if (m_marginals[m][mapped_indices[m]] < 0)
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

void sampleOne(size_t dimOffset, const std::vector<uint32_t>& subSeq, const NDArray<int64_t>& marginal, MappedIndex& index)
{
  static const double scale = 0.5 / (1u<<31);
  
  size_t dim = marginal.dim();

  if (dim > 2)
  {
    // reduce/fix if necessary, then slice the last dimension and recurse 
    // TODO this is wrong, all fixed dims should be sliced first. Need to track unfixed dims not just offset

    if (index[dimOffset + dim - 1] < 0)
    {
      const std::vector<int64_t>& r = reduce<int64_t>(marginal, dim - 1);
        // if all unfixed, reduce and recurse
      index[dimOffset + dim - 1] = pick(r.data(), r.size(), subSeq[dimOffset + dim - 1] * scale);
    }  
    const NDArray<int64_t>& sliced = slice(marginal, {dim - 1, index[dimOffset + dim - 1]});
    sampleOne(0, subSeq, sliced, index); 

    // TODO...
    // // reduce dim D-1
    // const std::vector<int64_t>& m = reduce(marginal, dim - 1);
    // // pick an index
    // int64_t oldValue = index[dimOffset + dim - 1];
    // index[dimOffset + dim-1] = pick(m.data(), m.size(), subSeq[dim-1] * scale);
    // if (oldValue != -1 && oldValue != index[dimOffset + dim - 1])
    // {
    //   std::cout << "warning: index change" << std::endl;
    // }

    // // take slice of Dim D-1 at index
    // const NDArray<int64_t>& sliced = slice(marginal, {dim-1, index[dim-1]});

    // // recurse
    // sampleOne(0, subSeq, sliced, index);
  }
  else if (dim == 2)
  {
    if (index[dimOffset] > -1 && index[dimOffset + 1] > -1)
    {
      return;
    }
    else if (index[dimOffset] > -1) // 2nd index needs to be fixed
    {
      const NDArray<int64_t>& sliced = slice(marginal, {0, index[dimOffset]});
      index[dimOffset + 1] = pick(sliced.rawData(), sliced.storageSize(), subSeq[dimOffset + 1] * scale);      
    }
    else if (index[dimOffset + 1] > -1) // 1st index needs to be fixed
    {
      // take slice in fixed dim
      const NDArray<int64_t>& sliced = slice(marginal, {1, index[dimOffset + 1]});
      index[dimOffset] = pick(sliced.rawData(), sliced.storageSize(), subSeq[dimOffset] * scale);
    }
    else //both indices need fixing
    {
      // reduce dim 1 (now 0)
      const std::vector<int64_t>& r1 = reduce<int64_t>(marginal, 1);
      // pick an index
      index[dimOffset + 1] = pick(r1.data(), r1.size(), subSeq[dimOffset + 1] * scale);

      // slice dim 2 (now 0)
      const NDArray<int64_t>& sliced = slice(marginal, {1, index[dimOffset + 1]});
      assert(sliced.dim() == 1);
      // no reduction required
      // pick an index
      index[dimOffset] = pick(sliced.rawData(), sliced.storageSize(), subSeq[dimOffset] * scale);
    }
  }
  else // 1-D
  {
    if (index[dimOffset] < 0)
    {
      index[dimOffset] = pick(marginal.rawData(), marginal.storageSize(), subSeq[dimOffset] * scale);
    }
  }

}

void QIS::sample(const std::vector<uint32_t>& seq, size_t marginalNo, MappedIndex& index)
{
  // pick relevant numbers from sample
  std::vector<uint32_t> r;
  for (size_t m = 0; m < m_indices[marginalNo].size(); ++m)
  {
    r.push_back(seq[m_indices[marginalNo][m]]);
  }
  // std::cout << marginalNo << ": ";
  // print(r);

  sampleOne(0, r, m_marginals[marginalNo], index);

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

