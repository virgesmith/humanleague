
#pragma once

// TODO can this be moved into cpp?
extern "C"
{
#include "SobolImpl.h"
}

#include <vector>
#include <limits>
#include <cstdint>
#if __cplusplus <= 201703l
#include <cstddef>
#endif

// This class is roughly compatible with C++11's distribution objects
// NB check for 32 vs 64 bit issues (distribution may expect 64 bit variates, this class returns 32bit)
class Sobol
{
public:

  static constexpr double SCALE = 1.0 / (1ull << std::numeric_limits<uint32_t>::digits);

  explicit Sobol(size_t dim, uint32_t nSkip = 0u);

  Sobol(const Sobol&) = delete;
  Sobol& operator=(const Sobol&) = delete;

  ~Sobol();

  const std::vector<uint32_t>& buf();

  // NB use with care in std::distribution objects, which may be expecting a 64-bit variate
  uint32_t operator()();

  // Skip largest 2^k <= n
  void skip(uint32_t n);

  void reset(uint32_t nSkip = 0u);

  uint32_t dim() const;

  uint32_t min() const;

  uint32_t max() const;

private:

  SobolData* m_s;
  uint32_t m_dim;
  std::vector<uint32_t> m_buf;
  uint32_t m_pos;
};

