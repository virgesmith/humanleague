
#include "GQIWS.h"

#include "Index.h"
#include "Sobol.h"
#include "DDWR.h"

//#define NO_R

// #ifdef NO_R
// #include <iostream>
// #define OSTREAM std::cout
// #else
// #include <Rcpp.h>
// #define OSTREAM Rcpp::Rcout
// #endif

// private helper functions for CQIWS
namespace {

bool constraintMet(const NDArray<bool>& allowed, QIWS::table_t& t)
{
  for (Index index(t.sizes()); !index.end(); ++index)
  {
    if (!allowed[index] && t[index])
    return false;
  }
  return true;
}

bool switchOne(const Index& forbiddenIndex, const NDArray<bool>& allowedStates, QIWS::table_t& pop)
{
  // TODO why 1000?
  if (pop[forbiddenIndex] > 1000) return true;

  // TODO randomise starting index
  std::vector<int64_t> switchFromIndex(2);
  size_t offset0 = std::rand() % pop.sizes()[0];
  size_t offset1 = std::rand() % pop.sizes()[1];
  //print(pop.rawData(), pop.storageSize(), pop.sizes()[1], Rcout);
  //Rcout << "Forbidden state is " << forbiddenIndex[0] << ", " << forbiddenIndex[1] << " value " << pop[forbiddenIndex] << std::endl;
  //Rcout << "Starting indices are " << offset0 << ", " << offset1 << std::endl;
  bool haveCachedSwitchState = false;
  std::vector<int64_t> cachedSwitchFromIndex(2);
  for (int64_t counter0 = 0; counter0 < pop.sizes()[0]; ++counter0)
  {
    for (int64_t counter1 = 0; counter1 < pop.sizes()[1]; ++counter1)
    {
      switchFromIndex[0] = (counter0 + offset0) % pop.sizes()[0];
      switchFromIndex[1] = (counter1 + offset1) % pop.sizes()[1];
      // must have different row and column indices to forbiddenIndex
      if (switchFromIndex[0] == forbiddenIndex[0] || switchFromIndex[1] == forbiddenIndex[1])
        continue;

      std::vector<int64_t> switchToIndexA{ forbiddenIndex[0], switchFromIndex[1] };
      std::vector<int64_t> switchToIndexB{ switchFromIndex[0], forbiddenIndex[1] };

      // Prefer if both switch-to states are allowed
      if ((allowedStates[switchToIndexA] && allowedStates[switchToIndexB])
            && pop[switchFromIndex]/*only needs to have 1 >= pop[forbiddenIndex]*/)
      {
        // Rcout << "Found pairable state at " << switchFromIndex[0] << ", " << switchFromIndex[1] << std::endl;
        // Rcout << "Found allowed state at " << switchToIndexA[0] << ", " << switchToIndexA[1] << std::endl;
        // Rcout << "Found allowed state at " << switchToIndexB[0] << ", " << switchToIndexB[1] << std::endl;

        // one at a time
        --pop[switchFromIndex];// -= pop[forbiddenIndex];
        ++pop[switchToIndexA];// += pop[forbiddenIndex];
        ++pop[switchToIndexB];// += pop[forbiddenIndex];
        --pop[forbiddenIndex];// = 0u;
        //print(pop.rawData(), pop.storageSize(), pop.sizes()[1], Rcout);
        return true;
      }
      //but also keep track of one place where a non-optimal switch can be made
      else if (!haveCachedSwitchState /*&& (!allowedStates[switchToIndexA] && !allowedStates[switchToIndexB])*/
                 && pop[switchFromIndex])
      {
        cachedSwitchFromIndex[0] = switchFromIndex[0];
        cachedSwitchFromIndex[1] = switchFromIndex[1];
        haveCachedSwitchState = true;
      }
    }
  }
  if (haveCachedSwitchState)
  {
    std::vector<int64_t> switchToIndexA{ forbiddenIndex[0], cachedSwitchFromIndex[1] };
    std::vector<int64_t> switchToIndexB{ cachedSwitchFromIndex[0], forbiddenIndex[1] };
    // Rcout << "Found pairable state at " << cachedSwitchFromIndex[0] << ", " << cachedSwitchFromIndex[1] << std::endl;
    // Rcout << "Found 1/2 allowed state at " << switchToIndexA[0] << ", " << switchToIndexA[1] << std::endl;
    // Rcout << "Found 1/2 allowed state at " << switchToIndexB[0] << ", " << switchToIndexB[1] << std::endl;
    --pop[cachedSwitchFromIndex];// -= pop[forbiddenIndex];
    ++pop[switchToIndexA];// += pop[forbiddenIndex];
    ++pop[switchToIndexB];// += pop[forbiddenIndex];
    --pop[forbiddenIndex];// = 0u;
    return true;
  }
  return false;
}

bool switchPop(const Index& forbiddenIndex, const NDArray<bool>& allowedStates, QIWS::table_t& pop)
{
  //print(pop.rawData(), pop.storageSize(), pop.sizes()[1], Rcout);
  //Rcout << "Forbidden state populated at " << forbiddenIndex[0] << ", " << forbiddenIndex[1] << std::endl;

  // we move pop from forbiddenIndex and switchFromIndex to switchToIndexA and switchToIndexB
  // which preserves marginals. The four indices form a rectangle

  while(pop[forbiddenIndex])
  {
    //Rcout << pop[forbiddenIndex] << std::endl;
    if (!switchOne(forbiddenIndex, allowedStates, pop))
      return false;
  }
  // notify if unable to switch
  return true;
}

// stat
ConstrainG::Status constrain(NDArray<uint32_t>& pop, const NDArray<bool>& allowedStates, const size_t iterLimit)
{
  size_t iter = 0;
  do
  {
    
    // Loop over all states, until no population in forbidden states
    for (Index idx(pop.sizes()); !idx.end(); ++idx)
    {
      if (!allowedStates[idx] && pop[idx])
      {
        if (!switchPop(idx, allowedStates, pop))
        {
          //throw std::runtime_error("unable to correct for forbidden states");
          return ConstrainG::STUCK;
        }
      }
    }
    ++iter;
  } while(iter < iterLimit && !constraintMet(allowedStates, pop));

  return iter == iterLimit ? ConstrainG::ITERLIMIT : ConstrainG::SUCCESS;
}


}

// TODO >2 dims
class DynamicSampler
{
public:
  DynamicSampler(const std::vector<std::vector<uint32_t>>& marginals, const NDArray<double>& exoProbs)
    : m_exoProbs(exoProbs), m_p(exoProbs.sizes())
  {
    for (size_t i = 0; i < marginals.size(); ++i)
    {
      m_dists.push_back(std::vector<int32_t>(marginals[i].begin(), marginals[i].end()));
    }
  }

  DynamicSampler(const DynamicSampler&) = delete;

  bool sample(size_t n, Sobol& sobol, NDArray<uint32_t>& pop)
  {
    m_marginalIntegrity = true;
    for (size_t i = 0; i < n; ++i)
    {
      // if (!sampleImpl(sobol, pop))
      //   return false;
      sampleImpl(sobol, pop);
    }
    //print(m_dists[0], OSTREAM);
    //print(m_dists[1], OSTREAM);
    return m_marginalIntegrity;
  }


private:

  void sampleImpl(Sobol& sobol, NDArray<uint32_t>& pop)
  {
    std::vector<double> m0(m_dists[0].size(), 0.0);
    std::vector<double> m1(m_dists[1].size(), 0.0);
    // update dynamic probabilities (assuming if we get stuck we remain stuck)
    m_marginalIntegrity &= update(m0, m1);
    // OSTREAM << "m0: "; print(m_dists[0], OSTREAM);
    // OSTREAM << "m1: "; print(m_dists[1], OSTREAM);
    // OSTREAM << "MI: " << m_marginalIntegrity << std::endl;
    // OSTREAM << "p0: "; print(m0, OSTREAM);
    // //OSTREAM << "p1: "; print(m1, OSTREAM);
    // print(m_p.rawData(), m_p.storageSize(), m1.size(), OSTREAM);

    // sample
    discrete_distribution_with_replacement<double> t0(m0.begin(), m0.end());

    std::vector<int64_t> idx(2);

    // sample dim 0
    uint32_t r = sobol();
    idx[0] = t0(r);
    //OSTREAM << r * 0.5/(1u<<31) << "->" << idx[0] << std::endl;

    // Drop marginal constraints if we end up in am impossible situation (will correct later)
    if (m_marginalIntegrity)
    {
      // update m1 for selected given index of m0
      for (idx[1] = 0; idx[1] < (int64_t)m_dists[1].size(); ++idx[1])
      {
        m1[idx[1]] = m_p[idx];
      }
    }
    else
    {
    // // check sum(m1) and bale if zero
    // if (std::accumulate(m1.begin(), m1.end(), 0.0) == 0.0)
    // {
      //m_marginalIntegrity = false;
      //
      //return false;
      // this needs to be consistent with the update function
      // for (idx[1] = 0; idx[1] < m_dists[1].size(); ++idx[1])
      // {
      //   m1[idx[1]] = m_exoProbs[idx];
      // }
      std::copy(m_dists[1].begin(), m_dists[1].end(), m1.begin());
    }
    //OSTREAM << "p1: "; print(m1,OSTREAM);

    discrete_distribution_with_replacement<double> t1(m1.begin(), m1.end());
    r = sobol();
    idx[1] = t1(r);
    //OSTREAM << r * 0.5/(1u<<31) << "->" << idx[1] << std::endl;

    //OSTREAM << idx[0] << "," << idx[1] << std::endl;
    //if (m_marginalIntegrity)
    {
      --m_dists[0][idx[0]];
      --m_dists[1][idx[1]];
    }
    ++pop[idx];
    //return m_marginalIntegrity;
  }


  // TODO just return m0 (m1 is not worth computing at this stage)
  bool update(std::vector<double>& m0, std::vector<double>& m1)
  {
    std::vector<int64_t> idx(2);
    // multiply joint dist from current marginal freqs by exogenous probabilities
    double sum = 0.0;
    for (idx[0] = 0; idx[0] < (int64_t)m_dists[0].size(); ++idx[0])
    {
      for (idx[1] = 0; idx[1] < (int64_t)m_dists[1].size(); ++idx[1])
      {
        // marginals can go -ve, need these as zero probabilities
        m_p[idx] = m_exoProbs[idx] * std::max(0,m_dists[0][idx[0]]) * std::max(0,m_dists[1][idx[1]]);
        sum += m_p[idx];
        m0[idx[0]] += m_p[idx];
        m1[idx[1]] += m_p[idx];
      }
    }

    // marginal and exogenous constraints are now in conflict.
    // // Strategy 1: Use only exogenous constraint from now on (and fix later)
    // if (sum == 0.0)
    // {
    //   m_marginalIntegrity = false;
    //   for (idx[0] = 0; idx[0] < m_dists[0].size(); ++idx[0])
    //   {
    //     for (idx[1] = 0; idx[1] < m_dists[1].size(); ++idx[1])
    //     {
    //       // marginals can go -ve, need these as zero probabilities
    //       m_p[idx] = m_exoProbs[idx];
    //       sum += m_p[idx];
    //       m0[idx[0]] += m_p[idx];
    //       m1[idx[1]] += m_p[idx];
    //     }
    //   }
    // }
    // Strategy 2: Use only marginal constraints from now on (and fix later)
    if (sum == 0.0)
    {
      m_marginalIntegrity = false;
      for (idx[0] = 0; idx[0] < (int64_t)m_dists[0].size(); ++idx[0])
      {
        for (idx[1] = 0; idx[1] < (int64_t)m_dists[1].size(); ++idx[1])
        {
          // marginals can go -ve, need these as zero probabilities
          m_p[idx] = std::max(0,m_dists[0][idx[0]]) * std::max(0,m_dists[1][idx[1]]);
          sum += m_p[idx];
        }
      }
      std::copy(m_dists[0].begin(), m_dists[0].end(), m0.begin());
      std::copy(m_dists[1].begin(), m_dists[1].end(), m1.begin());
    }
    //renomalise probabilities
    for (idx[0] = 0; idx[0] < (int64_t)m_dists[0].size(); ++idx[0])
    {
      for (idx[1] = 0; idx[1] < (int64_t)m_dists[1].size(); ++idx[1])
      {
        m_p[idx] /= sum;
      }
    }
    for (idx[0] = 0; idx[0] < (int64_t)m_dists[0].size(); ++idx[0])
    {
      m0[idx[0]] /= sum;
    }
    for (idx[1] = 0; idx[1] < (int64_t)m_dists[1].size(); ++idx[1])
    {
      m1[idx[1]] /= sum;
    }
    return m_marginalIntegrity;
  }

public:
  std::vector<std::vector<int32_t>> m_dists;
  // dodgy ref storage at least efficient
  const NDArray<double>& m_exoProbs;
  // joint dist incl exo probs
  NDArray<double> m_p;
  // flag to indicate whether sampling has violated marginal constraints
  bool m_marginalIntegrity;
};


GQIWS::GQIWS(const std::vector<marginal_t>& marginals, const NDArray<double>& exoProbs)
  : QIWS(marginals), m_exoprobs(exoProbs)
{
  for (Index index(exoProbs.sizes()); !index.end(); ++index)
  {
    if (m_exoprobs[index] < 0.0 || m_exoprobs[index] > 1.0)
      throw std::runtime_error("invalid exogenous probability");
  }
}

bool GQIWS::solve()
{
  // only initialised on first call, ensures different population each time
  // will throw when it reaches 2^32 samples
  static Sobol sobol(m_dim/*, m_sum*/);
  //static std::mt19937 sobol(70858048);

  bool success = false;
  size_t iter = 0;
  const size_t limit = 1;
  while (!success && iter<limit)
  {
    m_t.assign(0);
    DynamicSampler sampler(m_marginals, m_exoprobs);

    success = sampler.sample(m_sum, sobol, m_t);
    ++iter;
  }

  // switch population out of zero-probability states
  if (!success)
  {
    NDArray<bool> permitted(m_t.sizes());

    for (Index index(permitted.sizes()); !index.end(); ++index)
    {
      permitted[index] = m_exoprobs[index] > 0.0;
    }
    //print(permitted.rawData(), permitted.storageSize(), m_marginals[1].size(), OSTREAM);

    success = (constrain(m_t, permitted, 5) == ConstrainG::SUCCESS);
  }

  // print(m_t.rawData(), m_t.storageSize(), m_marginals[1].size(), OSTREAM);
  //OSTREAM << iter << " iterations" << std::endl;
  return success;
}






