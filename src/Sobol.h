
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
  explicit Sobol(uint32_t dim, uint32_t nSkip = 0u);

  ~Sobol();

  const std::vector<uint32_t>& buf();

  // NB use with care in std::distribtion objects, which may be expecting a 64-bit variate
  uint32_t operator()();

  // Skip largest 2^k <= n
  void skip(uint32_t n);

  uint64_t min() const;

  uint64_t max() const;

private:

  SobolData* m_s;
  uint32_t m_dim;
  std::vector<uint32_t> m_buf;
  uint32_t m_pos;
};
