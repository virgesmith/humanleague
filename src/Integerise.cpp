
#include "Integerise.h"
#include "QISI.h"
#include "NDArrayUtils.h"
#include "Log.h"

#include <algorithm>
#include <numeric>
#include <iostream>

std::vector<int> integeriseMarginalDistribution(const std::vector<double>& p, int pop, double& mse)
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

  mse = 0.0;
  for(size_t i = 0; i < n; ++i)
  {
    mse += r[i] * r[i];
  }
  mse /= n;

  return f;
}



// TODO needs to be an object (derived from QISI) (marginals are references in base class)
std::unique_ptr<QISI> integerise_multidim(const NDArray<double>& seed)
{
  // construct 1-d integer marginals in each dim
  size_t dim = seed.dim();
  double fsum = sum(seed);

  std::cout << "dim=%% sum=%%"_s % dim % fsum << std::endl;

  // TODO check (close to) integer
  (void)fsum;
  
  QISI::index_list_t indices(dim); // 0..n-1
  // hack
  static QISI::marginal_list_t marginals(dim);

  for (size_t d = 0; d < dim; ++d)
  {
    const std::vector<double>& mf = reduce(seed, d);
    // TODO check (close to) integers
    indices[d] = {(int64_t)d};
    marginals[d].resize({(int64_t)mf.size()});
    std::cout << "%%: %% %% %%" % indices[d] % marginals[d].dim() % marginals[d].sizes() % mf << std::endl;
    for (size_t i = 0; i < mf.size(); ++i)
    {
      *(marginals[d].begin() + i) = int64_t(mf[i] + 0.5); 
    }
  }

  return std::unique_ptr<QISI>(new QISI(indices, marginals));
}


