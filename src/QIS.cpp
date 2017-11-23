
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

void recursive_pick(const NDArray<double>& p, const std::vector<uint32_t>& seq, Index& index, size_t dim)
{
  static const double scale = 0.5 / (1u<<31);

  const std::vector<double> r = reduce<double>(p, dim);
  index[dim] = pick<double>(r.data(), r.size(), seq[dim] * scale);

  const NDArray<double>& s = slice(p, {dim, index[dim]});

  if (dim == 1)
  {
    index[0] = pick<double>(s.rawData(), s.storageSize(), seq[0] * scale);
    return;
  }
  else
  {
    recursive_pick(s, seq, index, dim-1);
  }
}


void recursive_sample(std::vector<std::pair<int64_t, uint32_t>>& dims, const NDArray<int64_t>& marginal,
                      MappedIndex& index, std::map<int64_t, int64_t> slice_map)
{
  static const double scale = 0.5 / (1u<<31);

#ifdef VERBOSE
  std::cout << "recursive_sample: " << dims.size() << " of " << marginal.dim() << std::endl;
#endif

  // end recursion at 1 (cannot have a zero-D marginal)
  if (dims.size() == 1)
  {
    index[dims.back().first] = pick(marginal.rawData(), marginal.storageSize(), dims.back().second * scale);
#ifdef VERBOSE
    std::cout << "marginal (1d):";
    print(marginal.rawData(), marginal.storageSize());
    std::cout << "recursive_sample picked: D" << dims.back().first << ":" << index[dims.back().first] << std::endl;
#endif
    dims.pop_back();
    return;
  }
  else
  {
    const std::vector<int64_t>& r = reduce<int64_t>(marginal, slice_map[dims.back().first]);
    index[dims.back().first] = pick(r.data(), r.size(), dims.back().second * scale);
    const NDArray<int64_t>& sliced = slice(marginal, { slice_map[dims.back().first], index[dims.back().first] });
#ifdef VERBOSE
    std::cout << "marginal (>1d):";
    print(marginal.rawData(), marginal.storageSize());
    std::cout << "recursive_sample picked: D" << dims.back().first << "[" << slice_map[dims.back().first]<< "]" << ":" << index[dims.back().first] << std::endl;
    std::cout << "sliced marginal:";
    print(sliced.rawData(), sliced.storageSize());
#endif
    dims.pop_back();
    recursive_sample(dims, sliced, index, slice_map);
  }
}


void sample(std::vector<int64_t>& dims, const std::vector<uint32_t>& seq, const NDArray<int64_t>& marginal, MappedIndex& index)
{
#ifdef VERBOSE
  std::cout << "dims:";
  print(dims);
  std::cout << "seq:";
  print(seq);
#endif
  std::vector<std::pair<int64_t, int64_t>> dims_to_slice;
  std::vector<std::pair<int64_t, uint32_t>> dims_to_sample;
  // this tracks dimensions as the array is sliced
  std::map<int64_t, int64_t> slice_map;
  int64_t slice_index = 0;
  for (size_t d = 0; d < dims.size(); ++d)
  {
    // d can be wrong here, perhaps because index is the wrong way round?
    if (index[d] > -1)
    {
      dims_to_slice.push_back(std::make_pair(d, index[d]));
    }
    else
    {
      dims_to_sample.push_back(std::make_pair(d, seq[dims[d]]));
      slice_map[d] = slice_index;
      ++slice_index;
    }
  }
#ifdef VERBOSE
  std::cout << "slice:";
  for (size_t i = 0; i < dims_to_slice.size(); ++i)
  {
    std::cout << dims_to_slice[i].first << ":" << dims_to_slice[i].second << std::endl;
  }
  std::cout << "sample:";
  for (size_t i = 0; i < dims_to_sample.size(); ++i)
  {
    std::cout << dims_to_sample[i].first << ":" << dims_to_sample[i].second << std::endl;
  }
  std::cout << "remapped dims for sampling:";
  for (auto it = slice_map.begin(); it != slice_map.end(); ++it)
  {
    std::cout << it->first << "->" << it->second << std::endl;
  }
#endif

  // nothing to do if all dims already sampled
  if (dims_to_sample.empty())
    return;

  // first get slice in the free dimensions only
  const NDArray<int64_t>& free = slice(marginal, dims_to_slice);

#ifdef VERBOSE
  std::cout << "sliced [" << free.dim() << "] ";
  print(free.rawData(), free.storageSize());
  std::cout << "remapped dims to sample:";
  for (size_t i = 0; i < dims_to_sample.size(); ++i)
  {
    std::cout << dims_to_sample[i].first << ":" << dims_to_sample[i].second << std::endl;
  }
#endif

  // should now have an array with dim = dims_to_sample.size()
  recursive_sample(dims_to_sample, free, index, slice_map);
}

}

QIS::QIS(const index_list_t& indices, marginal_list_t& marginals, int64_t skips)
: Microsynthesis(indices, marginals), m_sobolSeq(m_dim), m_conv(false)
{
  m_sobolSeq.skip(skips);
  m_stateValues.resize(m_array.sizes());
  // compute initial state probabilities and keep a copy
  computeStateValues();
  NDArray<double>::copy(m_stateValues, m_expectedStateOccupancy);
  // scale up to get expected occupancy
  double scale = m_population / sum(m_stateValues);
  for (Index index(m_expectedStateOccupancy.sizes()); !index.end(); ++index)
    m_expectedStateOccupancy[index] *= scale;
#ifdef VERBOSE
  // if (sum(m_expectedStateOccupancy) != m_population)
  //   throw std::logic_error("mismatch in expected state occupancy" + std::to_string(sum(m_expectedStateOccupancy)) + " vs " + std::to_string(m_population));
  std::cout << "scaling factor = " << 1.0 / scale << std::endl;
#endif
}


const NDArray<int64_t>& QIS::solve(bool reset)
{
  // slow, but simpler - samples from expected values
  //return solve_p(reset);
  // fast, but complicated - slices and dices each marginal
  return solve_m(reset);
}

const NDArray<int64_t>& QIS::solve_p(bool reset)
{
  if (reset)
  {
    m_sobolSeq.reset();
  }

  m_conv = true;
  m_array.assign(0ll);

  Index main_index(m_array.sizes());

  std::vector<MappedIndex> mapped_indices;
  mapped_indices.reserve(m_marginals.size());
  for (size_t m = 0; m < m_marginals.size(); ++m)
  {
    mapped_indices.push_back(MappedIndex(main_index, m_indices[m]));
  }

  // loop over population
  for (int64_t i = 0; i < m_population; ++i)
  {
    const std::vector<uint32_t>& seq = m_sobolSeq.buf();

    recursive_pick(m_stateValues, seq, main_index, m_dim-1);

    ++m_array[main_index];

    for (size_t m = 0; m < m_marginals.size(); ++m)
    {
      --m_marginals[m][mapped_indices[m]];
      if (m_marginals[m][mapped_indices[m]] < 0)
        m_conv = false;
    }
#ifdef VERBOSE
    print(m_stateValues.rawData(), m_stateValues.storageSize(), m_stateValues.sizes()[0]);
#endif
    if (m_stateValues[main_index] < 1.0)
      updateStateValues(main_index, mapped_indices);
  }
  m_chiSq = ::chiSq(m_array, m_expectedStateOccupancy);

  m_pValue = ::pValue(dof(m_array.sizes()), m_chiSq).first;

  m_degeneracy = ::degeneracy(m_array);

  return m_array;
}

// control state of Sobol via arg?
// better solution? construct set of 1-d marginals and sample from these
const NDArray<int64_t>& QIS::solve_m(bool reset)
{
  if (reset)
  {
    m_sobolSeq.reset();
  }

  m_conv = true;
  // loop over population
  m_array.assign(0ll);

  Index main_index(m_array.sizes());

  std::vector<MappedIndex> mapped_indices = makeMarginalMappings(main_index);

  for (int64_t i = 0; i < m_population; ++i)
  {
#ifdef VERBOSE
    std::cout << "pop: " << i << std::endl;
#endif
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
      sample(m_indices[m], seq, m_marginals[m], mapped_indices[m]);
#ifdef VERBOSE
      print(main_index.operator const std::vector<int64_t, std::allocator<int64_t>> &());
#endif
    }
    // check we have sampled in every dim
    for (size_t d = 0; d < m_dim; ++d)
      if (main_index[d] < 0)
        throw std::runtime_error("sampling error, not all dims have been set");

    for (size_t m = 0; m < mapped_indices.size(); ++m)
    {
      --m_marginals[m][mapped_indices[m]];
      if (m_marginals[m][mapped_indices[m]] < 0)
        m_conv = false;
    }
    // increment pop
    ++m_array[main_index];
#ifdef VERBOSE
    print(m_array.rawData(), m_array.storageSize());
    std::cout << std::endl;
#endif
  }

#ifdef VERBOSE
  for (size_t m = 0; m < mapped_indices.size(); ++m)
  {
    print(m_marginals[m].rawData(),
          m_marginals[m].storageSize());
  }
#endif

  m_chiSq = ::chiSq(m_array, m_expectedStateOccupancy);

  m_pValue = ::pValue(dof(m_array.sizes()), m_chiSq).first;

  m_degeneracy = ::degeneracy(m_array);

  return m_array;
}


// Expected state occupancy
const NDArray<double>& QIS::expectation()
{
  return m_expectedStateOccupancy;
}

void QIS::computeStateValues()
{
  Index index_main(m_array.sizes());

  std::vector<MappedIndex> mappings = makeMarginalMappings(index_main);

  m_stateValues.assign(1.0);
  for (; !index_main.end(); ++index_main)
  {
    for (size_t k = 0; k < m_marginals.size(); ++k)
    {
      m_stateValues[index_main] *= m_marginals[k][mappings[k]];
    }
  }
}


void QIS::updateStateValues(const Index& position, const std::vector<MappedIndex>& mappings)
{
  double update = 1.0;
  for (size_t k = 0; k < m_marginals.size(); ++k)
  {
    update *= m_marginals[k][mappings[k]]; 
  }
  m_stateValues[position] = update;
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

