
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

int main()
{
  std::array<std::vector<double>, 2> m = {std::vector<double>{1,2,3}, 
                                          std::vector<double>{3,3}};

  size_t size[2] = { m[0].size(), m[1].size() };                                        

  NDArray<2, double> s(size);
  s.assign(1.0);

  IPF(s, m);

}
  
// private:
//   NDArray<2, double> m_result;
//   std::array<std::vector<double>, 2> m_errors;
  
// };