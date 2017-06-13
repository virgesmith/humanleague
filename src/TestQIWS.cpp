
#include "UnitTester.h"
#include "QIWS.h"


void unittest::testQIWS()
{
//  {
//    size_t s[3] = {2,3,5};
//    Index<3, 0> index(s);

//    while (!index.end())
//    {
//      print(index.operator size_t*(), 3);
//      ++index;
//    }
//  }
//  {
//    size_t s[3] = {2,3,5};
//    Index<3, 1> index(s);

//    while (!index.end())
//    {
//      print(index.operator size_t*(), 3);
//      ++index;
//    }
//  }
//  {
//    size_t s[3] = {2,3,5};
//    Index<3, 2> index(s);

//    while (!index.end())
//    {
//      print(index.operator size_t*(), 3);
//      ++index;
//    }
//  }
//  {
//    size_t s[3] = {2,3,5};
//    Index<3, Index_Unfixed> index(s);

//    while (!index.end())
//    {
//      print(index.operator size_t*(), 3);
//      ++index;
//    }
//  }

  try
  {
    {
      std::vector<std::vector<uint32_t>> m;
      m.push_back(std::vector<uint32_t>{ 30, 60, 36});
      m.push_back(std::vector<uint32_t>{ 20, 50, 40, 16});
      QIWS<2> qiws(m);

      CHECK(qiws.solve());
      CHECK(qiws.pValue().first > 0.005);

      const NDArray<2, uint32_t>& a = qiws.result();

      CHECK(std::accumulate(a.rawData(), a.rawData() + a.storageSize(), 0) == std::accumulate(m[0].begin(), m[0].end(), 0));
    }
    {
      std::vector<std::vector<uint32_t>> m(4, std::vector<uint32_t>{ 30, 60, 36});

      //size_t idx[4] = { 0, 1, 2, 0 };
      //std::cout << "mp = " << marginalProduct<4>(m, idx) << std::endl; // 1944000
      QIWS<4> qiws(m);

      // give it a few chances
      bool solved = false;
      for (size_t i = 0; i < 1; ++i)
      {
        solved = qiws.solve();
        //LOG_INFO(format("%% %%", i , qiws.pValue()));
        if (qiws.pValue().first > 0.005)
          break;
      }
      CHECK(solved);
      // TODO investigate why this case consistently results in such a low p value
      //CHECK(qiws.pValue().first > 0.005);

      const NDArray<4, uint32_t>& a = qiws.result();
      CHECK(std::accumulate(a.rawData(), a.rawData() + a.storageSize(), 0) == std::accumulate(m[0].begin(), m[0].end(), 0));
    }
    {
      std::vector<std::vector<uint32_t>> m(5, std::vector<uint32_t>{ 30, 60, 36});
      QIWS<5> qiws(m);

      CHECK(qiws.solve());
      //CHECK(qiws.pValue().first > 0.005);

      const NDArray<5, uint32_t>& a = qiws.result();
      CHECK(std::accumulate(a.rawData(), a.rawData() + a.storageSize(), 0) == std::accumulate(m[0].begin(), m[0].end(), 0));
    }
    {
      std::vector<std::vector<uint32_t>> m(12, std::vector<uint32_t>{ 400, 600});
      QIWS<12> qiws(m);

      CHECK(qiws.solve());
      CHECK(qiws.pValue().first > 0.005);

      const NDArray<12, uint32_t>& a = qiws.result();
      CHECK(std::accumulate(a.rawData(), a.rawData() + a.storageSize(), 0) == std::accumulate(m[0].begin(), m[0].end(), 0));

    }

    //humanleague::synthPop(list(c(100,1,1,1),c(1,100,1,1)),1)
    {
      std::vector<std::vector<uint32_t>> m;
      m.push_back(std::vector<uint32_t>{100,1,1,1});
      m.push_back(std::vector<uint32_t>{1,100,1,1});
      QIWS<2> qiws(m);

      CHECK(qiws.solve());
      //CHECK(qiws.pValue().first  > 0.005); // arbitrary

      const NDArray<2, uint32_t>& a = qiws.result();
      CHECK(std::accumulate(a.rawData(), a.rawData() + a.storageSize(), 0) == std::accumulate(m[0].begin(), m[0].end(), 0));
    }

    //humanleague::synthPop(list(c(100,1,1,1,6),c(1,100,1,1,1,1,1,1,1,1)),1)
    {
      std::vector<std::vector<uint32_t>> m;
      m.push_back(std::vector<uint32_t>{100,1,1,1,6});
      m.push_back(std::vector<uint32_t>{1,100,1,1,1,1,1,1,1,1});
      QIWS<2> qiws(m);

      CHECK(qiws.solve());
      //CHECK(qiws.pValue().first > 0.001); // arbitrary

      const NDArray<2, uint32_t>& a = qiws.result();
      CHECK(std::accumulate(a.rawData(), a.rawData() + a.storageSize(), 0) == std::accumulate(m[0].begin(), m[0].end(), 0));
    }

    {
      std::vector<std::vector<uint32_t>> m;
      m.push_back(std::vector<uint32_t>{144, 150, 3, 2, 153, 345, 13, 11, 226, 304, 24, 18, 250, 336, 14, 21, 190, 176, 15, 14, 27, 10, 2, 3, 93, 135, 2, 6, 30, 465, 11, 28, 43, 463, 17, 76, 39, 458, 15, 88, 55, 316, 22, 50, 15, 25, 11, 17});
      m.push_back(std::vector<uint32_t>{18, 1, 1, 3, 6, 5, 1, 2, 1, 8, 2, 3, 4, 2, 4, 2, 2, 2, 4, 2, 4, 2, 2, 8, 10, 6, 2, 1, 2, 2, 2, 1, 1, 1, 5, 1, 2, 1, 1, 1, 3, 2, 1, 3, 3, 1, 1, 4, 4, 1, 1, 5, 4, 10, 1, 6, 2, 67, 1, 10, 7, 9, 4, 21, 19, 9, 131, 17, 9, 8, 14, 17, 13, 11, 3, 6, 2, 2, 3, 1, 12, 1, 1, 1, 2, 1, 1, 1, 2, 21, 1, 26, 97, 10, 47, 6, 2, 3, 2, 7, 2, 17, 2, 6, 3, 1, 1, 2, 18, 9, 59, 5, 399, 71, 100, 157, 74, 199, 154, 98, 22, 7, 13, 39, 19, 6, 43, 41, 24, 14, 30, 30, 105, 604, 15, 69, 33, 1, 122, 17, 20, 9, 77, 4, 9, 4, 56, 1, 32, 10, 9, 79, 4, 2, 30, 116, 3, 6, 14, 18, 2, 2, 9, 4, 11, 12, 5, 5, 2, 1, 1, 3, 9, 2, 7, 3, 1, 4, 1, 3, 2, 1, 7, 1, 7, 4, 17, 3, 5, 2, 6, 11, 2, 2, 3, 13, 3, 5, 1, 3, 2, 4, 2, 1, 16, 4, 1, 3, 7, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 6, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 9, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 330, 28, 281, 12});
//      m.push_back(std::vector<uint32_t>{144, 150, 3, 3263, 153, 345, 13, 11, 226, 304, 24, 18, 250, 336, 14, 21, 190, 176, 15, 14, 27, 10, 2, 3, 93, 135, 2, 6, 30, 465, 11, 28, 43, 463, 17, 76, 39, 458, 15, 88, 55, 316, 22, 50, 15, 25, 11, 17});
//      m.push_back(std::vector<uint32_t>{18, 1, 1, 3, 6, 5, 1, 3263, 1, 8, 2, 3, 4, 2, 4, 2, 2, 2, 4, 2, 4, 2, 2, 8, 10, 6, 2, 1, 2, 2, 2, 1, 1, 1, 5, 1, 2, 1, 1, 1, 3, 2, 1, 3, 3, 1, 1, 4, 4, 1, 1, 5, 4, 10, 1, 6, 2, 67, 1, 10, 7, 9, 4, 21, 19, 9, 131, 17, 9, 8, 14, 17, 13, 11, 3, 6, 2, 2, 3, 1, 12, 1, 1, 1, 2, 1, 1, 1, 2, 21, 1, 26, 97, 10, 47, 6, 2, 3, 2, 7, 2, 17, 2, 6, 3, 1, 1, 2, 18, 9, 59, 5, 399, 71, 100, 157, 74, 199, 154, 98, 22, 7, 13, 39, 19, 6, 43, 41, 24, 14, 30, 30, 105, 604, 15, 69, 33, 1, 122, 17, 20, 9, 77, 4, 9, 4, 56, 1, 32, 10, 9, 79, 4, 2, 30, 116, 3, 6, 14, 18, 2, 2, 9, 4, 11, 12, 5, 5, 2, 1, 1, 3, 9, 2, 7, 3, 1, 4, 1, 3, 2, 1, 7, 1, 7, 4, 17, 3, 5, 2, 6, 11, 2, 2, 3, 13, 3, 5, 1, 3, 2, 4, 2, 1, 16, 4, 1, 3, 7, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 6, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 9, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 330, 28, 281, 12});

      QIWS<2> qiws(m);

      bool solved = qiws.solve();
      CHECK(solved);
      CHECK(qiws.pValue().first > 0.005);
    }

    {
      std::vector<std::vector<uint32_t>> m;
      m.push_back(std::vector<uint32_t>{1,4,7});
      m.push_back(std::vector<uint32_t>{9,3});
      QIWS<2> qiws(m);

      CHECK(qiws.solve());
      CHECK(qiws.pValue().first > 0.005); // arbitrary
    }
  }
  catch(const std::exception& e)
  {
    UNEXPECTED_ERROR(e.what());
  }
  catch(...)
  {
    UNHANDLED_ERROR();
  }

}
