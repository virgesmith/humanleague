
#pragma once

extern "C"
{
#include "SobolImpl.h"
}

#include <vector>

#include <cstdint>

// This class is compatible with C++11's distribution objects
// However since this is multidimensional, if you require different
// distributions in different dimensions, you will have to use ...

// NASTY alert! C++11 distribution objects will oversample if the width of the uniform <64 bits
// which renders the QRNG useless.
// Workaround is to make return type 64 bits and return the Sobol value as the 32 MSB
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


