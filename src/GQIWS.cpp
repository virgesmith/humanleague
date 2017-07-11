
#include "GQIWS.h"

//#define NO_R

// #ifdef NO_R
// #include <iostream>
// #define OSTREAM std::cout
// #else
// #include <Rcpp.h>
// #define OSTREAM Rcpp::Rcout
// #endif

// TODO >2 dims
class DynamicSampler
{
public:
  DynamicSampler(const std::vector<std::vector<uint32_t>>& marginals, const NDArray<2,double>& exoProbs)
    : m_exoProbs(exoProbs), m_p(exoProbs.sizes())
  {
    for (size_t i = 0; i < marginals.size(); ++i)
    {
      m_dists.push_back(discrete_distribution_without_replacement<uint32_t>(marginals[i].begin(), marginals[i].end()));
    }
  }

  DynamicSampler(const DynamicSampler&) = delete;

  bool sample(size_t n, Sobol& sobol, NDArray<2, uint32_t>& pop)
  {
    for (size_t i = 0; i < n; ++i)
    {
      if (!sampleImpl(sobol, pop))
        return false;
    }
    return true;
  }


private:

  bool sampleImpl(Sobol& sobol, NDArray<2, uint32_t>& pop)
  {
    std::vector<double> m0(m_dists[0].freq().size(), 0.0);
    std::vector<double> m1(m_dists[1].freq().size(), 0.0);
    // update dynamic probabilities
    if (!update(m0, m1)) return false;
    // OSTREAM << "m0: "; print(m_dists[0].freq(), OSTREAM);
    // OSTREAM << "m1: "; print(m_dists[1].freq(), OSTREAM);
    // OSTREAM << "p0: "; print(m0, OSTREAM);
    // //OSTREAM << "p1: "; print(m1, OSTREAM);
    // print(m_p.rawData(), m_p.storageSize(), m1.size(), OSTREAM);

    // sample
    discrete_distribution_with_replacement<double> t0(m0.begin(), m0.end());

    size_t idx[2];

    // sample dim 0
    uint32_t r = sobol();
    idx[0] = t0(r);
    //OSTREAM << r * 0.5/(1u<<31) << "->" << idx[0] << std::endl;

    // update m1 for selected given index of m0
    for (idx[1] = 0; idx[1] < m_dists[1].freq().size(); ++idx[1])
    {
      m1[idx[1]] = m_p[idx];
    }
    // check sum(m1) and bale if zero
    if (std::accumulate(m1.begin(), m1.end(), 0.0) == 0.0)
      return false;
    // OSTREAM << "p1: "; print(m1,OSTREAM);

    discrete_distribution_with_replacement<double> t1(m1.begin(), m1.end());
    r = sobol();
    idx[1] = t1(r);
    //OSTREAM << r * 0.5/(1u<<31) << "->" << idx[1] << std::endl;

    // OSTREAM << idx[0] << "," << idx[1] << std::endl;
    --m_dists[0][idx[0]];
    --m_dists[1][idx[1]];

    ++pop[idx];
    return true;
  }


  bool update(std::vector<double>& m0, std::vector<double>& m1)
  {
    size_t idx[2];
    // multiply joint dist from current marginal freqs by exogenous probabilities
    double sum = 0.0;
    for (idx[0] = 0; idx[0] < m_dists[0].freq().size(); ++idx[0])
    {
      for (idx[1] = 0; idx[1] < m_dists[1].freq().size(); ++idx[1])
      {
        m_p[idx] = m_exoProbs[idx] * m_dists[0].freq()[idx[0]] * m_dists[1].freq()[idx[1]];
        sum += m_p[idx];
        m0[idx[0]] += m_p[idx];
        m1[idx[1]] += m_p[idx];
      }
    }

    if (sum == 0.0)
      return false;
    // renomalise probabilities
    for (idx[0] = 0; idx[0] < m_dists[0].freq().size(); ++idx[0])
    {
      for (idx[1] = 0; idx[1] < m_dists[1].freq().size(); ++idx[1])
      {
        m_p[idx] /= sum;
      }
    }
    for (idx[0] = 0; idx[0] < m_dists[0].freq().size(); ++idx[0])
    {
      m0[idx[0]] /= sum;
    }
    for (idx[1] = 0; idx[1] < m_dists[1].freq().size(); ++idx[1])
    {
      m1[idx[1]] /= sum;
    }
    return true;
  }

public:
  std::vector<discrete_distribution_without_replacement<uint32_t>> m_dists;
  // dodgy ref storage at least efficient
  const NDArray<2,double>& m_exoProbs;
  // joint dist incl exo probs
  NDArray<2, double> m_p;
};


GQIWS::GQIWS(const std::vector<marginal_t>& marginals, const NDArray<2, double>& exoProbs)
  : QIWS<2>(marginals), m_exoprobs(exoProbs)
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

bool GQIWS::solve()
{
  // TODO wrap a lot of this away in a distribution class...

  // only initialised on first call, ensures different population each time
  // will throw when it reaches 2^32 samples
  static Sobol sobol(Dim/*, m_sum*/);
  //static std::mt19937 sobol(70858048);

  bool success = false;
  size_t iter = 0;
  const size_t limit = 20;
  while (!success && iter<limit)
  {
    m_t.assign(0);
    DynamicSampler sampler(m_marginals, m_exoprobs);

    success = sampler.sample(m_sum, sobol, m_t);
    ++iter;
  }

  // print(m_t.rawData(), m_t.storageSize(), m_marginals[1].size(), OSTREAM);
  //OSTREAM << iter << " iterations" << std::endl;
  return success;
}







