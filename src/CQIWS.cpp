
#include "CQIWS.h"

 #include <Rcpp.h>
 using Rcpp::Rcout;

// private helper functions for CQIWS
namespace {

bool constraintMet(const NDArray<2, bool>& allowed, QIWS<2>::table_t& t)
{
  size_t index[2];

  // Loop over all states, until no population in forbidden states
  for (index[0] = 0; index[0] < t.sizes()[0]; ++index[0])
    for (index[1] = 0; index[1] < t.sizes()[1]; ++index[1])
    {
      if (!allowed[index] && t[index])
        return false;
    }
    return true;
}

bool switchOne(size_t* forbiddenIndex, const NDArray<2, bool>& allowedStates, QIWS<2>::table_t& pop)
{
  if (pop[forbiddenIndex] > 1000) return true;

  // TODO randomise starting index
  size_t switchFromIndex[2];
  size_t offset0 = std::rand() % pop.sizes()[0];
  size_t offset1 = std::rand() % pop.sizes()[1];
  //print(pop.rawData(), pop.storageSize(), pop.sizes()[1], Rcout);
  //Rcout << "Forbidden state is " << forbiddenIndex[0] << ", " << forbiddenIndex[1] << " value " << pop[forbiddenIndex] << std::endl;
  //Rcout << "Starting indices are " << offset0 << ", " << offset1 << std::endl;
  bool haveCachedSwitchState = false;
  size_t cachedSwitchFromIndex[2];
  for (size_t counter0 = 0; counter0 < pop.sizes()[0]; ++counter0)
  {
    for (size_t counter1 = 0; counter1 < pop.sizes()[1]; ++counter1)
    {
      switchFromIndex[0] = (counter0 + offset0) % pop.sizes()[0];
      switchFromIndex[1] = (counter1 + offset1) % pop.sizes()[1];
      // must have different row and column indices to forbiddenIndex
      if (switchFromIndex[0] == forbiddenIndex[0] || switchFromIndex[1] == forbiddenIndex[1])
        continue;

      size_t switchToIndexA[2] = { forbiddenIndex[0], switchFromIndex[1] };
      size_t switchToIndexB[2] = { switchFromIndex[0], forbiddenIndex[1] };

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
    size_t switchToIndexA[2] = { forbiddenIndex[0], cachedSwitchFromIndex[1] };
    size_t switchToIndexB[2] = { cachedSwitchFromIndex[0], forbiddenIndex[1] };
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

bool switchPop(size_t* forbiddenIndex, const NDArray<2, bool>& allowedStates, QIWS<2>::table_t& pop)
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

}

// static member function
Constrain::Status CQIWS::constrain(NDArray<2, uint32_t>& pop, const NDArray<2, bool>& allowedStates, const size_t iterLimit)
{
  size_t iter = 0;
  size_t idx[2];
  do
  {
    // Loop over all states, until no population in forbidden states
    for (idx[0] = 0; idx[0] < pop.sizes()[0]; ++idx[0])
      for (idx[1] = 0; idx[1] < pop.sizes()[1]; ++idx[1])
      {
        if (!allowedStates[idx] && pop[idx])
        {
          if (!switchPop(idx, allowedStates, pop))
          {
            //throw std::runtime_error("unable to correct for forbidden states");
            return Constrain::STUCK;
          }
        }
      }
      ++iter;
  } while(iter < iterLimit && !constraintMet(allowedStates, pop));

  return iter == iterLimit ? Constrain::ITERLIMIT : Constrain::SUCCESS;
}



CQIWS::CQIWS(const std::vector<marginal_t>& marginals, const NDArray<2, bool>& permittedStates)
  : QIWS<2>(marginals), m_allowedStates(permittedStates)
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

bool CQIWS::solve()
{
  size_t iter;
  const size_t iterLimit = m_t.storageSize();

  // size_t idx[2];
  // // disallow idx[1] > idx[0]
  // for (idx[0] = 0; idx[0] < m_t.sizes()[0]; ++idx[0])
  //   for (idx[1] = idx[0] + 2; idx[1] < m_t.sizes()[1]; ++idx[1])
  //     m_allowedStates[idx] = false;

  Constrain::Status status;
  // this loop appears unnecessary (always succeeds on first attempt is contraint is valid)
  for (size_t k = 0; k < 1; ++k)
  {
    QIWS::solve();
    // constraining...
    status = constrain(m_t, m_allowedStates, iterLimit);

    //Rcout << "Attempt " << k << " returned " << status << std::endl;
    if (status < Constrain::SUCCESS)
      break;
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

  Index<Dim, Index_Unfixed> index(m_t.sizes());

  double scale = 1.0 / std::pow(m_sum, Dim-1);

  double psum = 0.0;
  while (!index.end())
  {
    // m is the mean population of this state
    double m = marginalProduct<Dim>(m_marginals, index) * scale;
    m_p[index] = m / m_sum * m_allowedStates[index];
    psum += m_p[index];
    //m_chi2 += (m_t[index] - m) * (m_t[index] - m) / m;
    ++index;
  }

  Index<Dim, Index_Unfixed> index2(m_p.sizes());
  while (!index2.end())
  {
    m_p[index2] /= psum;
    ++index2;
  }

  // indicate not converged if iterLimit reached or stuck
  if (status != Constrain::SUCCESS)
    return false;

  return allZero;
}




