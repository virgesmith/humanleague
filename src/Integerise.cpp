
#include "Integerise.h"

#include <algorithm>

std::vector<int> integeriseMarginalDistribution(const std::vector<double>& p, size_t pop, double& mse)
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
