
#include "src/IPF.h"
#include "src/QSIPF.h"
#include "src/NDArrayUtils.h"
#include "src/Index.h"
#include "src/Sobol.h"

#include <iostream>

void do2d()
{
  std::vector<std::vector<double>> m = {std::vector<double>{52, 48}, 
                                        std::vector<double>{10, 77, 13}};

  size_t size[2] = { m[0].size(), m[1].size() };                                        

  NDArray<2, double> s(size);
  s.assign(1.0);
  //Index<2,Index_Unfixed> index(s.sizes());
  //s[index] = 0.5;

  QSIPF<2> qsipf(s, m);

  auto e = qsipf.errors();
  print(e[0]);
  print(e[1]);
  std::cout << qsipf.conv() << ":" << qsipf.iters() << std::endl;
  print(qsipf.result().rawData(), qsipf.result().storageSize(), m[1].size());
  print(qsipf.sample().rawData(), qsipf.sample().storageSize(), qsipf.sample().sizes()[1]);
  print(reduce<2, uint32_t, 0>(qsipf.sample()));
  print(reduce<2, uint32_t, 1>(qsipf.sample()));
}


void do3d()
{
  std::vector<std::vector<double>> m = {std::vector<double>{52, 40, 4, 4}, 
                                        std::vector<double>{87, 10, 3},
                                        std::vector<double>{55, 15, 6,12, 12}};
  // std::vector<std::vector<double>> m = {std::vector<double>{32, 32, 32, 32}, 
  //                                       std::vector<double>{126, 1, 1},
  //                                       std::vector<double>{64, 64}};

  size_t size[3] = { m[0].size(), m[1].size(), m[2].size() };                                        

  NDArray<3, double> s(size);
  s.assign(1.0);
  Index<3, Index_Unfixed> i(s.sizes());
  s[i] = 1.0;

  QSIPF<3> qsipf(s, m);

  auto e = qsipf.errors();
  print(e[0]);
  print(e[1]);
  print(e[2]);
  // conv/iters/errors needs a rethink
  std::cout << qsipf.conv() << ":" << qsipf.iters() << std::endl;
  //print(qsipf.result().rawData(), qsipf.result().storageSize(), m[1].size());

  //print(pop.rawData(), pop.storageSize(), pop.sizes()[1]);
  print(qsipf.sample().rawData(), qsipf.sample().storageSize(), qsipf.sample().sizes()[1]);
  print(reduce<3, uint32_t, 0>(qsipf.sample()));
  print(reduce<3, uint32_t, 1>(qsipf.sample()));
  print(reduce<3, uint32_t, 2>(qsipf.sample()));
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

  QSIPF<4> qsipf(s, m);

  auto e = qsipf.errors();
  print(e[0]);
  print(e[1]);
  print(e[2]);
  print(e[3]);
  //std::cout << qsipf.conv() << ":" << qsipf.iters() << std::endl;
  print(qsipf.sample().rawData(), qsipf.sample().storageSize(), m[1].size());
  print(reduce<4, uint32_t, 0>(qsipf.sample()));
  print(reduce<4, uint32_t, 1>(qsipf.sample()));
  print(reduce<4, uint32_t, 2>(qsipf.sample()));
  print(reduce<4, uint32_t, 3>(qsipf.sample()));
}

int main()
{
  try
  {
    do2d();
    do3d();
    do4d();
  }
  catch(const std::exception& e)
  {
    std::cout << e.what() << std::endl;
  }
  catch(...)
  {
    std::cout << "unknown exception" << std::endl;
  }  
}