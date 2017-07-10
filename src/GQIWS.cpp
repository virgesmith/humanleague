
#include "GQIWS.h"

#include <Rcpp.h>


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
  // only initialised on first call, ensures different population each time
  // will throw when it reaches 2^32 samples
  static Sobol sobol(Dim, m_sum);
  //static std::mt19937 sobol(70858048);

  std::vector<discrete_distribution_without_replacement<uint32_t>> dists;
  for (size_t i = 0; i < Dim; ++i)
  {
    dists.push_back(discrete_distribution_without_replacement<uint32_t>(m_marginals[i].begin(), m_marginals[i].end()));
  }

  // construct joint dist
  NDArray<2, double> p(m_exoprobs.sizes());

  size_t idx[Dim];
  m_t.assign(0u);

  for (size_t j = 0; j < m_sum; ++j)
  {
    // multiply joint dist from current marginal freqs by exogenous probabilities
    double sum = 0.0;
    std::vector<double> m0(dists[0].freq().size(), 0.0);
    std::vector<double> m1(dists[1].freq().size(), 0.0);
    for (idx[0] = 0; idx[0] < dists[0].freq().size(); ++idx[0])
    {
      for (idx[1] = 0; idx[1] < dists[1].freq().size(); ++idx[1])
      {
        p[idx] = m_exoprobs[idx] * dists[0].freq()[idx[0]] * dists[1].freq()[idx[1]];
        sum += p[idx];
        m0[idx[0]] += p[idx];
        m1[idx[1]] += p[idx];
      }
    }
    for (idx[0] = 0; idx[0] < dists[0].freq().size(); ++idx[0])
    {
      for (idx[1] = 0; idx[1] < dists[1].freq().size(); ++idx[1])
      {
        p[idx] /= sum;
      }
    }

    // double f = std::accumulate(m0.begin(), m0.end(), 0.0);
    // for (idx[0] = 0; idx[0] < dists[0].freq().size(); ++idx[0])
    // {
    //   m0[idx[0]] /= f;
    // }
    // for (idx[1] = 0; idx[1] < dists[1].freq().size(); ++idx[1])
    // {
    //   m1[idx[1]] /= f;
    // }

    //print(p.rawData(), p.storageSize(), m_marginals[1].size(), Rcpp::Rcout);
    print(m0, Rcpp::Rcout);
    print(m1, Rcpp::Rcout);

    discrete_distribution_with_replacement<double> t0(m0.begin(), m0.end());

    idx[0] = t0(sobol);
    for (idx[1] = 0; idx[1] < dists[1].freq().size(); ++idx[1])
    {
      m1[idx[1]] = p[idx];
    }
    print(m1, Rcpp::Rcout);
    discrete_distribution_with_replacement<double> t1(m1.begin(), m1.end());
    idx[1] = t1(sobol);
    Rcpp::Rcout << idx[0] << "," << idx[1] << std::endl;
    --dists[0][idx[0]];
    --dists[1][idx[1]];
    ++m_t[idx];
  }

  //
  // for (size_t j = 0; j < m_sum; ++j)
  // {
  //   for (size_t i = 0; i < Dim; ++i)
  //   {
  //     idx[i] = dists[i](sobol);
  //   }
  //   //print(idx, Dim);
  //   ++m_t[idx];
  // }
  //
  // std::vector<std::vector<int32_t>> r(Dim);
  // calcResiduals<Dim>(r);
  //
  // bool allZero = true;
  // for (size_t i = 0; i < Dim; ++i)
  // {
  //   int32_t m = maxAbsElement(r[i]);
  //   m_residuals[i] = m;
  //   allZero = allZero && (m == 0);
  // }
  //
  // m_chi2 = 0.0;
  //
  // Index<D, Index_Unfixed> index(m_t.sizes());
  //
  // double scale = 1.0 / std::pow(m_sum, Dim-1);
  //
  // while (!index.end())
  // {
  //   // m is the mean population of this state
  //   double m = marginalProduct<Dim>(m_marginals, index) * scale;
  //   m_p[index] = m / m_sum;
  //   m_chi2 += (m_t[index] - m) * (m_t[index] - m) / m;
  //   ++index;
  // }

  return true;
}







