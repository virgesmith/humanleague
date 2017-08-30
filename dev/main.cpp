
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
  Index<2,Index_Unfixed> index(s.sizes());
  s[index] = 0.5;

  IPF<2> ipf(s, m);

  auto e = ipf.errors();
  print(e[0]);
  print(e[1]);
  std::cout << ipf.conv() << ":" << ipf.iters() << std::endl;
}

// void do3d()
// {
//   std::array<std::vector<double>, 3> m = {std::vector<double>{52, 48}, 
//                                           std::vector<double>{87, 13},
//                                           std::vector<double>{45, 55}};

//   size_t size[3] = { m[0].size(), m[1].size(), m[2].size() };                                        

//   NDArray<3, double> s(size);
//   s.assign(1.0);
//   //Index<3,Index_Unfixed> index(s.sizes());
//   //s[index] = 0.5;

//   IPF<3> ipf(s, m);

//   auto e = ipf.errors();
//   print(e[0]);
//   print(e[1]);
//   print(e[2]);
//   std::cout << ipf.conv() << ":" << ipf.iters() << std::endl;
// }

int main()
{
  do2d();

}