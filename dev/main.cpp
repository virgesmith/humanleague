
#include "src/IPF.h"
#include "src/NDArrayUtils.h"
#include "src/Index.h"
#include "src/QIWS.h"

#include <iostream>

void do2d()
{
  std::vector<std::vector<double>> m = {std::vector<double>{52, 48}, 
                                        std::vector<double>{87, 13}};

  size_t size[2] = { m[0].size(), m[1].size() };                                        

  NDArray<2, double> s(size);
  s.assign(1.0);
  //Index<2,Index_Unfixed> index(s.sizes());
  //s[index] = 0.5;

  IPF<2> ipf(s, m);

  auto e = ipf.errors();
  print(e[0]);
  print(e[1]);
  std::cout << ipf.conv() << ":" << ipf.iters() << std::endl;
  print(ipf.result().rawData(), ipf.result().storageSize(), m[1].size());
}

void do3d()
{
  std::vector<std::vector<double>> m = {std::vector<double>{52, 48}, 
                                          std::vector<double>{87, 13},
                                          std::vector<double>{55, 45}};

  size_t size[3] = { m[0].size(), m[1].size(), m[2].size() };                                        

  NDArray<3, double> s(size);
  s.assign(1.0);
  // Index<3,Index_Unfixed> index(s.sizes());
  // s[index] = 0.7;

  IPF<3> ipf(s, m);

  auto e = ipf.errors();
  print(e[0]);
  print(e[1]);
  print(e[2]);
  std::cout << ipf.conv() << ":" << ipf.iters() << std::endl;
  print(ipf.result().rawData(), ipf.result().storageSize(), m[1].size());
}

void do4d()
{
  std::vector<std::vector<double>> m = {std::vector<double>{52, 48}, 
                                        std::vector<double>{87, 13},
                                        std::vector<double>{67, 33},
                                        std::vector<double>{55, 45}};

  size_t size[4] = { m[0].size(), m[1].size(), m[2].size(), m[3].size() };                                        

  NDArray<4, double> s(size);
  s.assign(1.0);
  //Index<3,Index_Unfixed> index(s.sizes());
  //s[index] = 0.7;

  IPF<4> ipf(s, m);

  auto e = ipf.errors();
  print(e[0]);
  print(e[1]);
  print(e[2]);
  print(e[3]);
  std::cout << ipf.conv() << ":" << ipf.iters() << std::endl;
  print(ipf.result().rawData(), ipf.result().storageSize(), m[1].size());
}

int main()
{
  do2d();
  do3d();
  do4d();
}