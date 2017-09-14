
#include "src/IPF.h"
#include "src/QSIPF.h"
#include "src/NDArrayUtils.h"
#include "src/Index.h"
#include "src/Sobol.h"

#include <iostream>

// delete this impl
size_t pick(const std::vector<double>& dist, double r)
{
  // sum of dist should be 1, but we relax this
  // r is in (0,1) so scale up r by sum of dist
  r *= std::accumulate(dist.begin(), dist.end(), 0.0);
  double runningSum = 0.0;
  for (size_t i = 0; i < dist.size(); ++i)
  {
    runningSum += dist[i];
    if (r < runningSum)
      return i;
  }
  throw std::runtime_error("pick failed");

}


void do2d()
{
  std::vector<std::vector<double>> m = {std::vector<double>{52, 48}, 
                                        std::vector<double>{10, 77, 13}};

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
  // std::vector<std::vector<double>> m = {std::vector<double>{52, 40, 4, 4}, 
  //                                       std::vector<double>{87, 10, 3},
  //                                       std::vector<double>{55, 15, 6,12, 12}};
  std::vector<std::vector<double>> m = {std::vector<double>{32, 32, 32, 32}, 
                                        std::vector<double>{126, 1, 1},
                                        std::vector<double>{64, 64}};

  size_t size[3] = { m[0].size(), m[1].size(), m[2].size() };                                        

  NDArray<3, double> s(size);
  s.assign(1.0);

  IPF<3> ipf(s, m);

  auto e = ipf.errors();
  print(e[0]);
  print(e[1]);
  print(e[2]);
  std::cout << ipf.conv() << ":" << ipf.iters() << std::endl;
  print(ipf.result().rawData(), ipf.result().storageSize(), m[1].size());

  // Sample without replacement of static IPF 
  size_t n = ipf.population();
  NDArray<3, double>& pop = const_cast<NDArray<3, double>&>(ipf.result());
  NDArray<3, uint32_t> sample(pop.sizes());
  sample.assign(0);
  Sobol qrng(3);
  const double scale = 0.5 / (1u<<31); 
  for (size_t i = 0; i < n; ++i)
  {
    const std::vector<uint32_t>& r = qrng.buf();
    size_t index[3] = {0};
    //for (size_t i = 0; i < n; ++i)

    // reduce dim 0
    const std::vector<double>& r0 = reduce<3, double, 0>(pop);
    // pick an index
    index[0] = pick(r0, r[0] * scale);

    // take slice Dim 0/index 
    NDArray<2, double> slice0 = slice<3, double, 0>(pop, index[0]);
    // reduce dim 1 (now 0)
    const std::vector<double>& r1 = reduce<2, double, 0>(slice0);
    // pick an index
    index[1] = pick(r1, r[1] * scale);

    // slice dim 2 (now 0)
    const std::vector<double>& r2 = slice<double, 0>(slice0, index[1]);
    // no reduction required
    // pick an index
    index[2] = pick(r2, r[2] * scale);

    // without replacement
    pop[index] = std::max(pop[index] - 1.0, 0.0);

    ++sample[index];
    //print(index, 3);
  }

  //print(pop.rawData(), pop.storageSize(), pop.sizes()[1]);
  print(sample.rawData(), sample.storageSize(), sample.sizes()[1]);
  print(reduce<3, uint32_t, 0>(sample));
  print(reduce<3, uint32_t, 1>(sample));
  print(reduce<3, uint32_t, 2>(sample));
 
}

void do3d_dynamic()
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
  s[i] = 0.7;

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
  //do2d();
  //do3d();
  do3d_dynamic();
  //do4d();
}