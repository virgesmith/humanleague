
#include "src/NDArray.h"
#include "src/GQIWS.h"

int main()
{
  std::vector<std::vector<uint32_t>> m;
  
  m.push_back({1, 3, 7, 19, 96, 4, 5, 1, 1});
  m.push_back({0, 7, 21, 109, 0, 0});
//  m.push_back({3, 2, 1});
//  m.push_back({2, 2, 1, 1});

  size_t s[2] = { m[0].size(), m[1].size() };

  NDArray<2, double> ep(s);
  ep.assign(1);
//  s[0]=0;s[1]=2; ep[s] = 0.0;
//  s[0]=0;s[1]=3; ep[s] = 0.0;
//  s[0]=1;s[1]=3; ep[s] = 0.0;
  print(ep.rawData(), ep.storageSize(), m[1].size(), std::cout);

  GQIWS gqiws(m, ep);
  
  std::cout << gqiws.solve() << std::endl;

}
