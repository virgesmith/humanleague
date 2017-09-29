
#if 0

#include "RQIWS.h"

RQIWS::RQIWS(const std::vector<marginal_t>& marginals, double rho)
  : QIWS<2>(marginals), m_rho(rho)
{
  if (m_marginals.size() != Dim)
  {
    throw std::runtime_error("invalid no. of marginals");
  }

  // check for -ve values have to loop manually and convert to signed value :(
  for (size_t i = 0; i < m_marginals.size(); ++i)
  {
    for (size_t j = 0; j < m_marginals[i].size(); ++j)
    {
      if (static_cast<int32_t>(m_marginals[i][j]) < 0)
        throw std::runtime_error("negative marginal value in marginal " + std::to_string(i) + " element " + std::to_string(j));
    }
  }

  size_t sizes[Dim];
  m_sum = sum(m_marginals[0]);
  sizes[0] = m_marginals[0].size();
  m_dof = sizes[0] - 1;
  for (size_t i = 1; i < m_marginals.size(); ++i)
  {
    if (m_sum != sum(m_marginals[i]))
    {
      throw std::runtime_error("invalid marginals");
    }
    sizes[i] = m_marginals[i].size();
    m_dof *= sizes[i] - 1;
  }
  m_t.resize(&sizes[0]);
  m_p.resize(&sizes[0]);

}

bool RQIWS::solve()
{
  // only initialised on first call, ensures different population each time
  // will throw when it reaches 2^32 samples
  static Sobol sobol(2, m_sum);
  //static std::mt19937 sobol(70858048);

  std::vector<discrete_distribution_without_replacement<uint32_t>> dists;
  dists.push_back(discrete_distribution_without_replacement<uint32_t>(m_marginals[0].begin(), m_marginals[0].end()));
  dists.push_back(discrete_distribution_without_replacement<uint32_t>(m_marginals[1].begin(), m_marginals[1].end()));

  m_t.assign(0u);

  Cholesky cholesky(m_rho);

  size_t idx[2];
  for (size_t j = 0; j < m_sum; ++j)
  {
    const std::pair<uint32_t, uint32_t>& cbuf = cholesky(sobol.buf());
    idx[0] = dists[0](cbuf.first);
    idx[1] = dists[1](cbuf.second);

    //print(idx, Dim);
    ++m_t[idx];
  }

  std::vector<std::vector<int32_t>> r(Dim);
  calcResiduals<Dim>(r);

  bool allZero = true;
  for (size_t i = 0; i < Dim; ++i)
  {
    int32_t m = maxAbsElement(r[i]);
    m_residuals[i] = m;
    allZero = allZero && (m == 0);
  }

  m_chi2 = 0.0;

  Index<2, Index_Unfixed> index(m_t.sizes());

  double scale = 1.0 / std::pow(m_sum, Dim-1);

  // TODO how to adjust p?
  while (!index.end())
  {
    // m is the mean population of this state
    double m = marginalProduct<Dim>(m_marginals, index) * scale;
    m_p[index] = m / m_sum;
    //m_chi2 += (m_t[index] - m) * (m_t[index] - m) / m;
    ++index;
  }

  return allZero;
}

#endif


