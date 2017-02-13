
#pragma once

extern "C"
{
#include "SobolImpl.h"
}

#include <vector>

#include <cstdint>

// This class is roughly compatible with C++11's distribution objects
// NB check for 32 vs 64 bit issues (distribution may expect 64 bit variates, this class returns 32bit)
class Sobol
{
public:

  typedef uint32_t result_type;

  explicit Sobol(uint32_t dim, result_type nSkip = 0u);

  ~Sobol();

  const std::vector<result_type>& buf();

  // NB use with care in std::distribtion objects, which may be expecting a 64-bit variate
  result_type operator()();

  // Skip largest 2^k <= n
  void skip(result_type n);

  result_type min() const;

  result_type max() const;

private:

  SobolData* m_s;
  uint32_t m_dim;
  std::vector<result_type> m_buf;
  uint32_t m_pos;
};
