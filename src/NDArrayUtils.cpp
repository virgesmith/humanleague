
#include "NDArrayUtils.h"

int32_t maxAbsElement(const std::vector<int32_t>& r)
{
  int32_t m = 0;
  for (size_t i = 0; i < r.size(); ++i)
  {
    m = std::max(m, abs(r[i]));
  }
  return m;
}

std::vector<int32_t> diff(const std::vector<uint32_t>& x, const std::vector<uint32_t>& y)
{
  size_t size = x.size();
  assert(size == y.size());

  std::vector<int32_t> result(size);

  for (size_t i = 0; i < x.size(); ++i)
  {
    result[i] = x[i] - y[i];
  }
  return result;
}

std::vector<double> diff(const std::vector<double>& x, const std::vector<double>& y)
{
  size_t size = x.size();
  assert(size == y.size());

  std::vector<double> result(size);

  for (size_t i = 0; i < x.size(); ++i)
  {
    result[i] = x[i] - y[i];
  }
  return result;
}


bool allZeros(const std::vector<std::vector<int32_t>>& r)
{
  for (const auto& v: r)
    if (!isZero(v))
      return false;
    return true;
}

std::vector<std::vector<int>> listify(const size_t pop, const NDArray<uint32_t>& t)
{
  std::vector<std::vector<int>> list(t.dim(), std::vector<int>(pop));
  Index index(t.sizes());

  size_t pindex = 0;
  while (!index.end())
  {
    for (size_t i = 0; i < t[index]; ++i)
    {
      const std::vector<int64_t>& ref = index;
      for (size_t j = 0; j < t.dim(); ++j)
      {
        list[j][pindex] = ref[j];
      }
      ++pindex;
    }
    ++index;
  }
  return list;
}


