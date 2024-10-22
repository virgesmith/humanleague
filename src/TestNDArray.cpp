
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

  for (idx[0] = 0; idx[0] < (int64_t)a.size(0); ++idx[0])
  {
    for (idx[1] = 0; idx[1] < (int64_t)a.size(1); ++idx[1])
    {
      for (idx[2] = 0; idx[2] < (int64_t)a.size(2); ++idx[2])
      {
        uint32_t n = idx[0] * 100 + idx[1] * 10 + idx[2];
        a[idx] = n;
      }
    }
  }
}

