
#include "UnitTester.h"
#include "NDArray.h"
#include "Index.h"

#include <cstdint>

void unittest::testNDArray()
{
  std::vector<int64_t> s{5,2,3};

  NDArray<uint32_t> a(s);

  CHECK(a.size(0) == 5);
  CHECK(a.size(1) == 2);
  CHECK(a.size(2) == 3);

  Index idx(a.sizes());

  for (idx[0] = 0; idx[0] < a.size(0); ++idx[0])
  {
    for (idx[1] = 0; idx[1] < a.size(1); ++idx[1])
    {
      for (idx[2] = 0; idx[2] < a.size(2); ++idx[2])
      {
        uint32_t n = idx[0] * 100 + idx[1] * 10 + idx[2];
        //std::cout << idx[0] << idx[1] << idx[2] << "->";
        a[idx] = n;
        //std::cout << a[idx] << std::endl;
      }
    }
  }

//  {
//    NDArray<3, uint32_t>::ConstIterator<0> it(a, v);
//    std::cout << it.idx()[0] << it.idx()[1] << it.idx()[2] << std::endl;
//  }
//  {
//    NDArray<3, uint32_t>::ConstIterator<1> it(a, v);
//    std::cout << it.idx()[0] << it.idx()[1] << it.idx()[2] << std::endl;
//  }
//  {
//    NDArray<3, uint32_t>::ConstIterator<2> it(a, v);
//    std::cout << it.idx()[0] << it.idx()[1] << it.idx()[2] << std::endl;
//  }

  // {
  //   //std::cout << "dir0" << std::endl;
  //   size_t v[3] = { 0, a.size(1)-1, a.size(2)-1 };
  //   old::NDArray<3, uint32_t>::ConstIterator<0> it(a, v);
  //   CHECK(*it == 12);
  //   CHECK(*++it == 112);
  //   CHECK(*++it == 212);
  //   CHECK(*++it == 312);
  //   CHECK(*++it == 412);
  // }
  // {
  //   //std::cout << "dir1" << std::endl;
  //   size_t v[3] = { a.size(0)-1, 0, a.size(2)-1 };
  //   old::NDArray<3, uint32_t>::ConstIterator<1> it(a, v);
  //   CHECK(*it == 402);
  //   CHECK(*++it == 412);
  // }
  // {
  //   //std::cout << "dir2" << std::endl;
  //   size_t v[3] = { a.size(0)-1, a.size(1)-1, 0 };
  //   old::NDArray<3, uint32_t>::ConstIterator<2> it(a, v);
  //   CHECK(*it == 410);
  //   CHECK(*++it == 411);
  //   CHECK(*++it == 412);
  // }
  //
  // size_t v[3] = {1, 1, 1};
  // old::NDArray<3, uint32_t>::Iterator<0> it(a,v);
  // *it = 5;
  // {
  //   old::NDArray<3, uint32_t>::ConstIterator<0> it(a, v);
  //   CHECK(*it == 5);
  //   CHECK(*++it == 111);
  //   CHECK(*++it == 211);
  //   CHECK(*++it == 311);
  //   CHECK(*++it == 411);
  // }
  //
  // std::vector<size_t> m(20,2); // 2^20 elements = ~4MB
  // old::NDArray<10, uint32_t> multid(&m[0]);
};

