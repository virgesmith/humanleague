
#include "Integerise.h"
#include "QISI.h"
#include "NDArrayUtils.h"
#include "Log.h"

#include <algorithm>
#include <numeric>
#include <cmath>
//#include <iostream>

namespace {

int64_t checked_round(double x, double tol=1e-4) // loose tolerance ~1/4 mantissa precision
{
  if (fabs(x - round(x)) > tol)
    throw std::runtime_error("Marginal or total value %% is not an integer (within tolerance %%)"_s % x % tol);
  return (int64_t)round(x);
}

}

std::vector<int> integeriseMarginalDistribution(const std::vector<double>& p, int pop, double& rmse)
{
  const size_t n = p.size();
  std::vector<int> f(n);
  std::vector<double> r(n);

  for(size_t i = 0; i < n; ++i)
  {
    f[i] = p[i] * pop; // rounded down
    r[i] = p[i] * pop - f[i];
  }

  while(std::accumulate(f.begin(), f.end(), 0) < pop)
  {
    // find max
    auto it = max_element(r.begin(), r.end());
    ++*(f.begin() + std::distance(r.begin(), it));
    --*it;
  }

  rmse = 0.0;
  for (size_t i = 0; i < n; ++i)
  {
    rmse += r[i] * r[i];
  }
  rmse = sqrt(rmse / n);

  return f;
}

Integeriser::Integeriser(const NDArray<double>& seed) : m_seed(seed)
{
  // construct 1-d integer marginals in each dim
  size_t dim = m_seed.dim();
  // check total population is integral (or close)
  checked_round(sum(m_seed));

  // 1-d special case: use prob2IntFreq
  if (dim == 1) 
  {
    // convert to vector (reduce 1-d special case)
    std::vector<double> p = reduce(seed, 0);
    int pop = sum(seed);
    // convert to probabilities
    for (auto& x: p) x /= pop;
    std::vector<int> tmp = integeriseMarginalDistribution(p, pop, m_rmse);
    m_result.resize({(int64_t)tmp.size()});
    std::copy(tmp.begin(), tmp.end(), m_result.begin());
    m_conv = true;
    return;
  }

  m_indices.resize(dim); // 0..n-1
  m_marginals.resize(dim);

  for (size_t d = 0; d < dim; ++d)
  {
    const std::vector<double>& mf = reduce(seed, d);
    // TODO check (close to) integers
    m_indices[d] = {(int64_t)d};
    m_marginals[d].resize({(int64_t)mf.size()});
    //std::cout << "%%: %% %% %%" % m_indices[d] % m_marginals[d].dim() % m_marginals[d].sizes() % mf << std::endl;
    for (size_t i = 0; i < mf.size(); ++i)
    {
      *(m_marginals[d].begin() + i) = checked_round(mf[i]);
    }
  }

  QISI qisi(m_indices, m_marginals);

  NDArray<int64_t>::copy(qisi.solve(m_seed), m_result);
  m_conv = qisi.conv();

  m_rmse = 0.0;
  for (Index index(m_result.sizes()); !index.end(); ++index)
  {
    m_rmse += (m_result[index] - m_seed[index]) * (m_result[index] - m_seed[index]);
  }
  m_rmse = sqrt(m_rmse / m_result.storageSize());
}

const NDArray<int64_t>& Integeriser::result() const
{
  return m_result;
}

bool Integeriser::conv() const
{
  return m_conv;
}

double Integeriser::rmse() const
{
  return m_rmse;
}



