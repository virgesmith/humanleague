
#include "src/IPF.h"
#include "src/QIS.h"
//#include "src/QSIPF.h"
#include "src/QIS.h"
#include "src/QISI.h"
//#include "src/Microsynthesis.h"
#include "src/Sobol.h"

// // deprecated
// #include "src/NDArrayUtilsOld.h"
// #include "src/IndexOld.h"

// work-in-progress
#include "src/NDArray.h"
#include "src/Index.h"
#include "src/NDArrayUtils.h"


// void do2d()
// {
//   std::vector<std::vector<int64_t>> m = {std::vector<int64_t>{52, 48}, 
//                                         std::vector<int64_t>{10, 77, 13}};
                                    
//   std::vector<int64_t> size{ (int64_t)m[0].size(), (int64_t)m[1].size() };                                        

//   NDArray<double> s(size);
//   s.assign(1.0);
//   //Index index(s.sizes());
//   //s[index] = 0.5;

//   QSIPF qsipf(s, m);

//   auto e = qsipf.errors();
//   print(e[0]);
//   print(e[1]);
//   std::cout << qsipf.conv() << ":" << qsipf.iters() << ":" << qsipf.chiSq() << std::endl;
//   print(qsipf.result().rawData(), qsipf.result().storageSize(), m[1].size());
//   print(qsipf.sample().rawData(), qsipf.sample().storageSize(), qsipf.sample().sizes()[1]);
//   print(reduce<int64_t>(qsipf.sample(), 0));
//   print(reduce<int64_t>(qsipf.sample(), 1));
//}

// void do2dIPF()
// {
//   std::vector<std::vector<double>> m = {std::vector<double>{12, 40, 48}, 
//                                         std::vector<double>{87, 13}};
                                    
//   std::vector<int64_t> size{ (int64_t)m[0].size(), (int64_t)m[1].size() };                                        

//   NDArray<double> s(size);
//   s.assign(1.0);
//   //Index index(s.sizes());
//   //s[index] = 0.5;

//   IPF ipf(s, m);

//   auto e = ipf.errors();
//   print(e[0]);
//   print(e[1]);
//   std::cout << ipf.conv() << ":" << ipf.iters() << std::endl;
//   print(ipf.result().rawData(), ipf.result().storageSize(), m[1].size());
//   print(reduce(ipf.result(), 0));
//   print(reduce(ipf.result(), 1));
// }

// void do3d()
// {
//   std::vector<std::vector<int64_t>> m = {std::vector<int64_t>{52, 40, 4, 4}, 
//                                         std::vector<int64_t>{87, 10, 3},
//                                         std::vector<int64_t>{55, 15, 6,12, 12}};
//   // std::vector<std::vector<double>> m = {std::vector<int64_t>{32, 32, 32, 32}, 
//   //                                       std::vector<int64_t>{126, 1, 1},
//   //                                       std::vector<int64_t>{64, 64}};

//   std::vector<int64_t> size{ (int64_t)m[0].size(), (int64_t)m[1].size(), (int64_t)m[2].size() };                                        

//   NDArray<double> s(size);
//   s.assign(1.0);
//   Index i(s.sizes());
//   s[i] = 1.0;

//   QSIPF qsipf(s, m);

//   auto e = qsipf.errors();
//   print(e[0]);
//   print(e[1]);
//   print(e[2]);
//   // conv/iters/errors needs a rethink
//   std::cout << qsipf.conv() << ":" << qsipf.iters() << ":" << qsipf.chiSq() << std::endl;
//   //print(qsipf.result().rawData(), qsipf.result().storageSize(), m[1].size());

//   //print(pop.rawData(), pop.storageSize(), pop.sizes()[1]);
//   print(qsipf.sample().rawData(), qsipf.sample().storageSize(), qsipf.sample().sizes()[1]);
//   print(reduce<int64_t>(qsipf.sample(), 0));
//   print(reduce<int64_t>(qsipf.sample(), 1));
//   print(reduce<int64_t>(qsipf.sample(), 2));
// }

// void do3dIPF()
// {
//   std::vector<std::vector<double>> m = {std::vector<double>{52, 48}, 
//                                         std::vector<double>{10, 77, 13},
//                                         std::vector<double>{20, 27, 30, 23}};
  
//   std::vector<int64_t> size{ (int64_t)m[0].size(), (int64_t)m[1].size(), (int64_t)m[2].size() };                                        

//   NDArray<double> s(size);
//   s.assign(1.0);
//   //Index index(s.sizes());
//   //s[index] = 0.5;

//   IPF ipf(s, m);

//   auto e = ipf.errors();
//   print(e[0]);
//   print(e[1]);
//   print(e[2]);
//   std::cout << ipf.conv() << ":" << ipf.iters() << std::endl;
//   print(ipf.result().rawData(), ipf.result().storageSize(), m[1].size());
//   print(reduce(ipf.result(), 0));
//   print(reduce(ipf.result(), 1));
//   print(reduce(ipf.result(), 2));
// }

// void do4d()
// {
//   std::vector<std::vector<int64_t>> m = {std::vector<int64_t>{52, 48}, 
//                                         std::vector<int64_t>{87, 13},
//                                         std::vector<int64_t>{67, 33},
//                                         std::vector<int64_t>{55, 45}};

//   std::vector<int64_t> size{ (int64_t)m[0].size(), (int64_t)m[1].size(), (int64_t)m[2].size(), (int64_t)m[3].size() };                                        

//   NDArray<double> s(size);
//   s.assign(1.0);
//   //Index index(s.sizes());
//   //s[index] = 0.7;

//   QSIPF qsipf(s, m);

//   auto e = qsipf.errors();
//   print(e[0]);
//   print(e[1]);
//   print(e[2]);
//   print(e[3]);
//   std::cout << qsipf.conv() << ":" << qsipf.iters() << ":" << qsipf.chiSq() << std::endl;
//   print(qsipf.sample().rawData(), qsipf.sample().storageSize(), m[1].size());
//   print(reduce<int64_t>(qsipf.sample(), 0));
//   print(reduce<int64_t>(qsipf.sample(), 1));
//   print(reduce<int64_t>(qsipf.sample(), 2));
//   print(reduce<int64_t>(qsipf.sample(), 3));
// }

void doMd()
{
  std::vector<NDArray<double>> m;
  std::vector<std::vector<int64_t>> i;
  //i.push_back(std::vector<int64_t>{0,1});
  //i.push_back(std::vector<int64_t>{2,1});
  //i.push_back(std::vector<int64_t>{2,0});
  i.push_back(std::vector<int64_t>{0});
  i.push_back(std::vector<int64_t>{1});
  NDArray<double> m0(std::vector<int64_t>{2});
  Index i0(m0.sizes());
  m0[i0] = 52.0;
  m0[++i0] = 48.0;
  //m0.assign(5.0);
  NDArray<double> m1(std::vector<int64_t>{2});
  Index i1(m1.sizes());
  m1[i1] = 87.0;
  m1[++i1] = 13.0;
  //m1.assign(2.0);
  //NDArray<double> m2(std::vector<int64_t>{5, 2});
  //m2.assign(9.0);
  // induce partial sum mismatch
  // Index idx(m1.sizes());
  // m1[idx] += 0.1;
  // m1[++idx] -= 0.1;

  m.push_back(std::move(m0));
  m.push_back(std::move(m1));
  //m.push_back(std::move(m2));
  // wip::QIS qis(i, m); // needs int marginals
  // {
  //   const auto& a = mipf.doP();
  //   print(a.rawData(), a.storageSize());
  //   std::cout << sum(a) << std::endl;
  // }
  IPF<double> mipf(i, m);
  {
    NDArray<double> s({m[0].sizes()[0], m[1].sizes()[0]});
    s.assign(1.0);
    const auto& a = mipf.solve(s);
    print(a.rawData(), a.storageSize());
    std::cout << sum(a) << std::endl;
  }
}

void doMd_QIS()
{
  std::vector<NDArray<int64_t>> m;
  std::vector<std::vector<int64_t>> i;
  i.push_back(std::vector<int64_t>{0,1});
  i.push_back(std::vector<int64_t>{1,2});
  NDArray<int64_t> m0(std::vector<int64_t>{2,2});
  Index i0(m0.sizes());
  m0[i0] = 52;
  m0[++i0] = 48;
  //m0.assign(5.0);
  NDArray<int64_t> m1(std::vector<int64_t>{2});
  Index i1(m1.sizes());
  m1[i1] = 87;
  m1[++i1] = 13;

  m.push_back(std::move(m0));
  m.push_back(std::move(m1));

  QIS qis(i, m);
  {
    const auto& a = qis.solve();
    print(a.rawData(), a.storageSize());
    std::cout << sum(a) << std::endl;
  }
}

void doMd_QISI()
{
  std::vector<NDArray<int64_t>> m;
  std::vector<std::vector<int64_t>> i;
  i.push_back(std::vector<int64_t>{0});
  i.push_back(std::vector<int64_t>{1});
  NDArray<int64_t> m0(std::vector<int64_t>{3});
  Index i0(m0.sizes());
  m0[i0] = 52;
  m0[++i0] = 24;
  m0[++i0] = 24;
  //m0.assign(5.0);
  NDArray<int64_t> m1(std::vector<int64_t>{2});
  Index i1(m1.sizes());
  m1[i1] = 87;
  m1[++i1] = 13;
  NDArray<double> s({m0.sizes()[0], m1.sizes()[0]});
  s.assign(1.0);

  m.push_back(std::move(m0));
  m.push_back(std::move(m1));

  QISI qisi(i, m);
  {
    const auto& a = qisi.solve(s);
    print(a.rawData(), a.storageSize());
    std::cout << sum(a) << std::endl;
  }
}

int main()
{
  try
  {

    std::vector<int64_t> s{3,2,5};
    NDArray<double> a(s);
    a.assign(1.0);
    for (Index index(a.sizes(), std::make_pair(0,1)); !index.end(); ++index)
    {
      ++a[index];
    }
    for (Index index(a.sizes(), std::make_pair(1,0)); !index.end(); ++index)
    {
      ++a[index];
    }
    for (Index index(a.sizes(), std::make_pair(2,2)); !index.end(); ++index)
    {
      ++a[index];
    }
    print(a.rawData(), a.storageSize(), s[2]);

    print(reduce(a, 0));
    print(reduce(a, 1));
    print(reduce(a, 2));

    for (size_t d = 0; d < a.dim(); ++d)
    {
      for (int64_t i = 0; i < a.sizes()[d]; ++i)
      {
        NDArray<double> a00 = slice(a, {d,i});
        print(a00.rawData(), a00.storageSize());//, a00.sizes()[1]);
      }
    }

    {
      NDArray<double> r = reduce(a, std::vector<int64_t>{0,1});

      std::cout << r.dim() << std::endl;
      print(r.sizes());
      print(r.rawData(), r.storageSize(), r.sizes()[1]);
    }
    {
      NDArray<double> r = reduce(a, std::vector<int64_t>{1,2});

      std::cout << r.dim() << std::endl;
      print(r.sizes());
      print(r.rawData(), r.storageSize(), r.sizes()[1]);
    }
    {
      NDArray<double> r = reduce(a, std::vector<int64_t>{2, 0});

      std::cout << r.dim() << std::endl;
      print(r.sizes());
      print(r.rawData(), r.storageSize(), r.sizes()[1]);
    }

    //doMd();
    doMd_QIS();
    //doMd_QISI();
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