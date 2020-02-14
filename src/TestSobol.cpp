
#include "UnitTester.h"

#include "Sobol.h"
#include "DDWR.h"

#include <cstdint>
#include <algorithm>
#include <random>
#include <iostream>

namespace unittest {

void testSobol()
{
  {

    const size_t dim = 12;
    Sobol s(dim);
    CHECK_EQUAL(s.min(), 0u);
    CHECK_EQUAL(s.max(), -1u); // -1ul fails on LP64

    s.skip(12345);
    const std::vector<uint32_t>& b0 = s.buf();
    s.reset(12345);
    const std::vector<uint32_t>& b1 = s.buf();
    for (size_t i = 0; i < dim; ++i)
    {
      CHECK_EQUAL(b0[i], b1[i]);
    }
  }

  {


    std::vector<uint32_t> m{144, 150, 3, 2, 153, 345, 13, 11, 226, 304, 24, 18, 250, 336, 14, 21, 190, 176, 15, 14, 27, 10, 2, 3, 93, 135, 2, 6, 30, 465, 11, 28, 43, 463, 17, 76, 39, 458, 15, 88, 55, 316, 22, 50, 15, 25, 11, 17};

    uint32_t samples = std::accumulate(m.begin(), m.end(), 0);

    Sobol s(1);
    discrete_distribution_without_replacement<uint32_t> dist(m.begin(), m.end());

    std::vector<uint32_t> p(m.size(), 0);

    for (size_t i = 0; i < samples; ++i)
    {
      ++p[dist(s)];
    }

    for (size_t i = 0; i < m.size(); ++i)
    {
      CHECK(p[i] == m[i]);
    }

  }
}

}
