#include "Sobol.h"
#include "Log.h"

#include <limits>
#include <stdexcept>
#include <cstdint>


Sobol::Sobol(size_t dim, uint32_t nSkip) : m_dim(dim), m_buf(dim), m_pos(dim) // ensures m_buf gets populated on 1st access
{
  if (dim < 1 || dim > 1111)
  {
    throw std::range_error("Dim %% is not in valid range [1,1111]"s % dim);
  }

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
    throw std::out_of_range("Exceeded generation limit (2^32-1)");
  return m_buf;
}

uint32_t Sobol::operator()()
{
  if (m_pos == m_dim)
  {
    buf();
    m_pos = 0;
  }
  return m_buf[m_pos++];
}

// Skip largest 2^k-1 < n
void Sobol::skip(uint32_t n)
{
  uint32_t b = 0;
  while (n > 1)
  {
    ++b;
    n >>= 1;
  }

  uint32_t k = 1 << b;
  while (k--)
  {
    buf();
  }
}


void Sobol::reset(uint32_t nSkip)
{
  nlopt_sobol_destroy(m_s);
  m_s = nlopt_sobol_create(m_dim);
  if (nSkip > 0)
    skip(nSkip);
}

uint32_t Sobol::dim() const
{
  return m_dim;
}

uint32_t Sobol::min() const
{
  return 0;
}

uint32_t Sobol::max() const
{
  return std::numeric_limits<uint32_t>::max();
}


