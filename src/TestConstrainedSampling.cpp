

#include "UnitTester.h"
#include "CQIWS.h"
//#include "NDArrayUtils.h"


void unittest::testConstrainedSampling()
{
  // simple constraint: rooms >= beds
  {
    std::vector<uint32_t> rooms{ 1, 2, 3, 4};
    std::vector<uint32_t> beds{4, 3, 2, 1};
    std::vector<std::vector<uint32_t>> dists;
    dists.push_back(rooms);
    dists.push_back(beds);

    CQIWS cqiws(dists);

    CHECK(cqiws.solve());

  }
  {
    //                           1  2   3    4    5   6   7   8+
    std::vector<uint32_t> rooms{ 0, 3, 17, 124, 167, 79, 46, 22 };
    //                          1    2    3   4  5+
    std::vector<uint32_t> beds{15, 165, 238, 33, 7};

    std::vector<std::vector<uint32_t>> dists;
    dists.push_back(rooms);
    dists.push_back(beds);

    CQIWS cqiws(dists);

    CHECK(cqiws.solve());
  }
  {
    std::vector<uint32_t> rooms{ 1, 1, 1, 1, 1, 1, 1, 1 };
    std::vector<uint32_t> beds{2, 2, 2, 1, 1};
    std::vector<std::vector<uint32_t>> dists;
    dists.push_back(rooms);
    dists.push_back(beds);

    CQIWS cqiws(dists);

    CHECK(cqiws.solve());
  }
}
