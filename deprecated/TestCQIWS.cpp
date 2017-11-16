#if 0

#include "UnitTester.h"
#include "CQIWS.h"
//#include "NDArrayUtils.h"

#include <Rcpp.h>

void unittest::testConstrainedSampling()
{
  // simple constraint: rooms >= beds
  {
    std::vector<uint32_t> rooms{ 1, 2, 3, 4};
    std::vector<uint32_t> beds{4, 3, 2, 1};
    std::vector<std::vector<uint32_t>> dists;
    dists.push_back(rooms);
    dists.push_back(beds);

    size_t dim[2] = {rooms.size(), beds.size()};
    NDArray<2, bool> permitted(dim);
    permitted.assign(true);

    //Rcpp::Rcout << runCQIWS(dists) << std::endl;
    CQIWS cqiws(dists, permitted);
    CHECK(cqiws.solve());
  }
  // {
  //   //                           1  2   3    4    5   6   7   8+
  //   std::vector<uint32_t> rooms{ 0, 3, 17, 124, 167, 79, 46, 22 };
  //   //                          1    2    3   4  5+
  //   std::vector<uint32_t> beds{15, 165, 238, 33, 7};
  //
  //   std::vector<std::vector<uint32_t>> dists;
  //   dists.push_back(rooms);
  //   dists.push_back(beds);
  //
  //   //Rcpp::Rcout << runCQIWS(dists) << std::endl;
  //   CQIWS cqiws(dists);
  //   CHECK(cqiws.solve());
  // }
  // {
  //   std::vector<uint32_t> rooms{ 1, 1, 1, 1, 1, 1, 1, 1 };
  //   std::vector<uint32_t> beds{2, 2, 2, 1, 1};
  //   std::vector<std::vector<uint32_t>> dists;
  //   dists.push_back(rooms);
  //   dists.push_back(beds);
  //
  //   //Rcpp::Rcout << runCQIWS(dists) << std::endl;
  //   CQIWS cqiws(dists);
  //   CHECK(cqiws.solve());
  // }
  // {
  //   std::vector<uint32_t> rooms{10, 30, 48, 118, 24, 7, 3, 10};
  //   std::vector<uint32_t> beds{90, 125, 23, 5, 7};
  //   std::vector<std::vector<uint32_t>> dists;
  //   dists.push_back(rooms);
  //   dists.push_back(beds);
  //
  //   //Rcpp::Rcout << runCQIWS(dists) << std::endl;
  //   CQIWS cqiws(dists);
  //   CHECK(cqiws.solve());
  //
  // }
  // {
  //   std::vector<uint32_t> rooms{1, 1, 8, 3,84, 21, 4, 4, 1};
  //   std::vector<uint32_t> beds{0, 8, 3, 113, 2, 1};
  //   std::vector<std::vector<uint32_t>> dists;
  //   dists.push_back(rooms);
  //   dists.push_back(beds);
  //
  //   //Rcpp::Rcout << runCQIWS(dists) << std::endl;
  //   CQIWS cqiws(dists);
  //   CHECK(cqiws.solve());
  // }
}

#endif
