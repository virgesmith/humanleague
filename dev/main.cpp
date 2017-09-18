
#include "src/IPF.h"
#include "src/QSIPF.h"
#include "src/NDArrayUtils.h"
#include "src/Index.h"
#include "src/Sobol.h"

// work-in-progress
#include "src/NDArray2.h"
#include "src/Index2.h"
#include "src/NDArrayUtils2.h"
#include "src/IPF2.h"

#include <iostream>

// TODO integer marginals

void do2d()
{
  std::vector<std::vector<int64_t>> m = {std::vector<int64_t>{52, 48}, 
                                        std::vector<int64_t>{10, 77, 13}};
                                    
  size_t size[2] = { m[0].size(), m[1].size() };                                        

  NDArray<2, double> s(size);
  s.assign(1.0);
  //Index<2,Index_Unfixed> index(s.sizes());
  //s[index] = 0.5;

  QSIPF<2> qsipf(s, m);

  auto e = qsipf.errors();
  print(e[0]);
  print(e[1]);
  std::cout << qsipf.conv() << ":" << qsipf.iters() << ":" << qsipf.chiSq() << std::endl;
  print(qsipf.result().rawData(), qsipf.result().storageSize(), m[1].size());
  print(qsipf.sample().rawData(), qsipf.sample().storageSize(), qsipf.sample().sizes()[1]);
  print(reduce<2, int64_t, 0>(qsipf.sample()));
  print(reduce<2, int64_t, 1>(qsipf.sample()));
}

void do2dIPF()
{
  std::vector<std::vector<double>> m = {std::vector<double>{12, 40, 48}, 
                                        std::vector<double>{87, 13}};
                                    
  std::vector<int64_t> size{ (int64_t)m[0].size(), (int64_t)m[1].size() };                                        

  wip::NDArray<double> s(size);
  s.assign(1.0);
  //Index<2,Index_Unfixed> index(s.sizes());
  //s[index] = 0.5;

  wip::IPF ipf(s, m);

  auto e = ipf.errors();
  print(e[0]);
  print(e[1]);
  std::cout << ipf.conv() << ":" << ipf.iters() << std::endl;
  print(ipf.result().rawData(), ipf.result().storageSize(), m[1].size());
  print(wip::reduce(ipf.result(), 0));
  print(wip::reduce(ipf.result(), 1));
}


void do3d()
{
  std::vector<std::vector<int64_t>> m = {std::vector<int64_t>{52, 40, 4, 4}, 
                                        std::vector<int64_t>{87, 10, 3},
                                        std::vector<int64_t>{55, 15, 6,12, 12}};
  // std::vector<std::vector<double>> m = {std::vector<int64_t>{32, 32, 32, 32}, 
  //                                       std::vector<int64_t>{126, 1, 1},
  //                                       std::vector<int64_t>{64, 64}};

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
  std::cout << qsipf.conv() << ":" << qsipf.iters() << ":" << qsipf.chiSq() << std::endl;
  //print(qsipf.result().rawData(), qsipf.result().storageSize(), m[1].size());

  //print(pop.rawData(), pop.storageSize(), pop.sizes()[1]);
  print(qsipf.sample().rawData(), qsipf.sample().storageSize(), qsipf.sample().sizes()[1]);
  print(reduce<3, int64_t, 0>(qsipf.sample()));
  print(reduce<3, int64_t, 1>(qsipf.sample()));
  print(reduce<3, int64_t, 2>(qsipf.sample()));
}

void do3dIPF()
{
  std::vector<std::vector<double>> m = {std::vector<double>{52, 48}, 
                                        std::vector<double>{10, 77, 13},
                                        std::vector<double>{20, 27, 30, 23}};
  
  std::vector<int64_t> size{ (int64_t)m[0].size(), (int64_t)m[1].size(), (int64_t)m[2].size() };                                        

  wip::NDArray<double> s(size);
  s.assign(1.0);
  //Index<2,Index_Unfixed> index(s.sizes());
  //s[index] = 0.5;

  wip::IPF ipf(s, m);

  auto e = ipf.errors();
  print(e[0]);
  print(e[1]);
  print(e[2]);
  std::cout << ipf.conv() << ":" << ipf.iters() << std::endl;
  print(ipf.result().rawData(), ipf.result().storageSize(), m[1].size());
  print(wip::reduce(ipf.result(), 0));
  print(wip::reduce(ipf.result(), 1));
  print(wip::reduce(ipf.result(), 2));
}



void do4d()
{
  std::vector<std::vector<int64_t>> m = {std::vector<int64_t>{52, 48}, 
                                        std::vector<int64_t>{87, 13},
                                        std::vector<int64_t>{67, 33},
                                        std::vector<int64_t>{55, 45}};

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
  std::cout << qsipf.conv() << ":" << qsipf.iters() << ":" << qsipf.chiSq() << std::endl;
  print(qsipf.sample().rawData(), qsipf.sample().storageSize(), m[1].size());
  print(reduce<4, int64_t, 0>(qsipf.sample()));
  print(reduce<4, int64_t, 1>(qsipf.sample()));
  print(reduce<4, int64_t, 2>(qsipf.sample()));
  print(reduce<4, int64_t, 3>(qsipf.sample()));
}

int main()
{
  try
  {
    do2d();
    do3d();
    do4d();

    std::vector<int64_t> s{3,2,5};
    wip::NDArray<double> a(s);
    a.assign(1.0);
    for (wip::Index index(a.sizes(), {0,1}); !index.end(); ++index)
    {
      ++a[index];
    }
    for (wip::Index index(a.sizes(), {1,0}); !index.end(); ++index)
    {
      ++a[index];
    }
    for (wip::Index index(a.sizes(), {2,2}); !index.end(); ++index)
    {
      ++a[index];
    }
    print(a.rawData(), a.storageSize(), s[2]);

    print(wip::reduce(a, 0));
    print(wip::reduce(a, 1));
    print(wip::reduce(a, 2));

    for (size_t d = 0; d < a.dim(); ++d)
    {
      for (int64_t i = 0; i < a.sizes()[d]; ++i)
      {
        wip::NDArray<double> a00 = wip::slice(a, {d,i});
        print(a00.rawData(), a00.storageSize());//, a00.sizes()[1]);
      }
    }

    do2dIPF();
    do3dIPF();

    {
      wip::NDArray<double> r = reduce(a, std::vector<int64_t>{0,1});

      std::cout << r.dim() << std::endl;
      print(r.sizes());
      print(r.rawData(), r.storageSize(), r.sizes()[1]);
    }
    {
      wip::NDArray<double> r = reduce(a, std::vector<int64_t>{1,2});

      std::cout << r.dim() << std::endl;
      print(r.sizes());
      print(r.rawData(), r.storageSize(), r.sizes()[1]);
    }
    {
      wip::NDArray<double> r = reduce(a, std::vector<int64_t>{2, 0});

      std::cout << r.dim() << std::endl;
      print(r.sizes());
      print(r.rawData(), r.storageSize(), r.sizes()[1]);
    }
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