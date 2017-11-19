#include "UnitTester.h"

#include "NDArray.h"
#include "NDArrayUtils.h"

void unittest::testReduce()
{
  {
    int64_t values2[] = {0,1,2, 10,11,12};
    NDArray<int64_t> a2({2,3}, values2);

    const std::vector<int64_t>& r0 = reduce(a2, 0);
    CHECK_EQUAL(r0[0], 3);
    CHECK_EQUAL(r0[1], 33);
    const std::vector<int64_t>& r1 = reduce(a2, 1);
    CHECK_EQUAL(r1[0], 10);
    CHECK_EQUAL(r1[1], 12);
    CHECK_EQUAL(r1[2], 14);

    CHECK_THROWS(reduce(a2,2), std::runtime_error);
  }
  {
    int64_t values3[] = {0,1,2,3,4, 10,11,12,13,14, 20,21,22,23,24, 100,101,102,103,104, 110,111,112,113,114, 120,121,122,123,124};
    NDArray<int64_t> a3({2,3,5}, values3);
    const std::vector<int64_t>& r0 = reduce(a3, 0);
    CHECK_EQUAL(r0[0], 180);
    CHECK_EQUAL(r0[1], 1680);
    const std::vector<int64_t>& r1 = reduce(a3, 1);
    CHECK_EQUAL(r1[0], 520);
    CHECK_EQUAL(r1[1], 620);
    CHECK_EQUAL(r1[2], 720);
    const std::vector<int64_t>& r2 = reduce(a3, 2);
    CHECK_EQUAL(r2[0], 360);
    CHECK_EQUAL(r2[1], 366);
    CHECK_EQUAL(r2[2], 372);
    CHECK_EQUAL(r2[3], 378);
    CHECK_EQUAL(r2[4], 384);

    CHECK_THROWS(reduce(a3,-1), std::runtime_error);
    CHECK_THROWS(reduce(a3,17), std::runtime_error);
  }
}
