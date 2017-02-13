//

#include "Sobol.h"

#include <limits>
#include <stdexcept>
#include <cstdint>

#include <iostream>

Sobol::Sobol(uint32_t dim, uint32_t nSkip) : m_dim(dim), m_buf(dim), m_pos(dim) // ensures m_buf gets populated on 1st access
{
  m_s = nlopt_sobol_create(dim);
  if (nSkip > 0)
    skip(nSkip);
}

Sobol::~Sobol()
{
  nlopt_sobol_destroy(m_s);
}

const std::vector<uint32_t>& Sobol::buf()
{
  // TODO assert m_pos != m_dim|0 ? (i.e some of seq already used)
  if (!nlopt_sobol_next(m_s, &m_buf[0]))
    throw std::runtime_error("Exceeded generation limit (2^32-1)");
  return m_buf;
}

uint32_t Sobol::operator()()
{
  if (m_pos == m_dim)
  {
    if (!nlopt_sobol_next(m_s, &m_buf[0]))
      throw std::runtime_error("Exceeded generation limit (2^32-1)");
    m_pos = 0;
  }
  return m_buf[m_pos++];
}

// Skip largest 2^k <= n
void Sobol::skip(uint32_t n)
{
  uint32_t k = 1;
  while (k <= n)
    k *= 2;

  //std::cout << "skips=" << k << std::endl;
  uint32_t skipped = 0;
  while (--k > 0)
  {
    ++skipped;
    buf();
  }
  //std::cout << "skipped=" << skipped << std::endl;
}

uint32_t Sobol::min() const
{
  return 0;
}

uint32_t Sobol::max() const
{
  return std::numeric_limits<uint32_t>::max();
}


