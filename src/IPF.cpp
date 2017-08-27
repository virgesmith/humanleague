
#include "IPF.h"

#include "NDArrayUtils.h"

IPF::IPF(const NDArray<2, double>& seed, const std::array<std::vector<double>, 2>& marginals) : m_result(seed.sizes())
{
  print(marginals[0]);
  print(marginals[1]);
  print(reduce<2, double, 0>(seed));
  print(reduce<2, double, 1>(seed));

  std::copy(seed.rawData(), seed.rawData() + seed.storageSize(), m_result.begin());
  print(reduce<2, double, 0>(seed));
  print(reduce<2, double, 1>(seed));

}

  
// private:
//   NDArray<2, double> m_result;
//   std::array<std::vector<double>, 2> m_errors;
  
// };