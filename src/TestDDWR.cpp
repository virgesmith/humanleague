
#include "UnitTester.h"
#include "DDWR.h"
#include "Sobol.h"
#include <random>
#include <algorithm>

#include <iostream>

void unittest::testDDWR()
{
  try
  {
    {
      std::vector<uint32_t> m{ 30, 60, 36};
      std::vector<uint32_t> n{ 0, 0, 0};

      discrete_distribution_without_replacement<uint32_t> ddwor(m.begin(), m.end());

      Sobol s(1);
      uint32_t pop = std::accumulate(m.begin(), m.end(), 0);
      // populate n
      for (size_t i = 0; i < pop; ++i)
      {
        ++n[ddwor(s)];
      }
      // check n now matches m
      for (size_t i = 0; i < n.size(); ++i)
      {
        CHECK(n[i] == m[i]);
      }
    }

    {
      std::vector<uint32_t> m{144, 150, 3, 3263, 153, 345, 13, 11, 226, 304, 24, 18, 250, 336, 14, 21, 190, 176, 15, 14, 27, 10, 2, 3, 93, 135, 2, 6, 30, 465, 11, 28, 43, 463, 17, 76, 39, 458, 15, 88, 55, 316, 22, 50, 15, 25, 11, 17};
      std::vector<uint32_t> n(m.size(), 0);

      discrete_distribution_with_replacement <uint32_t> ddwir(m.begin(), m.end());

      uint32_t pop = std::accumulate(m.begin(), m.end(), 0);
      Sobol s(1,pop);


      // populate n
      for (size_t i = 0; i < pop; ++i)
      {
        ++n[ddwir(s)];
      }
      // check n now matches m (it should, since m_sum is a power of 2)
      for (size_t i = 0; i < n.size(); ++i)
      {
        CHECK(n[i] == m[i]);
      }
    }

    {
      std::vector<uint32_t> m{144, 150, 3, 2, 153, 345, 13, 11, 226, 304, 24, 18, 250, 336, 14, 21, 190, 176, 15, 14, 27, 10, 2, 3, 93, 135, 2, 6, 30, 465, 11, 28, 43, 463, 17, 76, 39, 458, 15, 88, 55, 316, 22, 50, 15, 25, 11, 17};
      std::vector<uint32_t> n(m.size(), 0);

      discrete_distribution_with_replacement <uint32_t> ddwir(m.begin(), m.end());

      uint32_t pop = std::accumulate(m.begin(), m.end(), 0);
      Sobol s(1,pop);

      // populate n
      for (size_t i = 0; i < pop; ++i)
      {
        ++n[ddwir(s)];
      }
      // check n now close to m
      for (size_t i = 0; i < n.size(); ++i)
      {
        CHECK(abs(n[i] - m[i]) < 3);
        //std::cout << m[i] << ": " << n[i] << std::endl;
      }
      CHECK(std::accumulate(n.begin(), n.end(), 0u) == pop);
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
