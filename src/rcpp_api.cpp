
#include "NDArrayUtils.h"
#include "Index.h"
#include "IPF.h"
#include "QIS.h"
#include "QISI.h"
#include "Integerise.h"
#include "StatFuncs.h"
#include "Sobol.h"

#include "UnitTester.h"

#include <Rcpp.h>

#include <vector>
#include <cstdint>

using namespace Rcpp;

//#include <csignal>

// // Handler for ctrl-C
// extern "C" void sigint_handler(int)
// {
//   // throw back to R
//   throw std::runtime_error("User interrupt");

// }
// Enable ctrl-C to interrupt the code
// TODO this doesnt seem to work, perhaps another approach (like a separate thread?)
//void (*oldhandler)(int) = signal(SIGINT, sigint_handler);

namespace Rhelpers {

template<typename T, typename R>
NDArray<T> convertArray(R rArray)
{
  // workaround for 1-d arrays (which don't have "dim" attribute)
  //Dimension colMajorSizes(rArray.hasAttribute("dim") ? Dimension(rArray.attr("dim")) : Dimension((size_t)rArray.size()));
  std::vector<int64_t> colMajorSizes;
  if (rArray.hasAttribute("dim"))
  {
    colMajorSizes = as<std::vector<int64_t>>(rArray.attr("dim"));
  }
  else
  {
    colMajorSizes.push_back(rArray.size());
  }
  const size_t dim = colMajorSizes.size();

  // This is column major data - reverse the dimensions but copy the data as-is for efficiency
  NDArray<T> array(colMajorSizes);

  // This makes IPF work correctly
  size_t i = 0;
  for (TransposedIndex idx(colMajorSizes); !idx.end(); ++idx, ++i)
  {
    array[idx] = rArray[i];
  }

  return array;
}

// Helper to get overall dimension and sizes before constructing QIS
std::vector<int64_t> getDimension(List indices, List marginals)
{
  std::map<int64_t,int64_t> lookup;
  // dont worry about inconsistencies here, QIS will detect and report them
  for (size_t i = 0; i < indices.size(); ++i)
  {
    const IntegerVector& iv = indices[i];
    const IntegerVector& mv = marginals[i];
    std::vector<int64_t> colMajorSizes;
    if (mv.hasAttribute("dim"))
    {
      colMajorSizes = as<std::vector<int64_t>>(mv.attr("dim"));
    }
    else
    {
      colMajorSizes.push_back(mv.size());
    }
    for (size_t j = 0; j < colMajorSizes.size(); ++j)
    {
      lookup[iv[j]] = colMajorSizes[j];
    }
  }
  std::vector<int64_t> ret;
  ret.reserve(lookup.size());
  for (const auto& kv: lookup)
    ret.push_back(kv.second);
  return ret;
}

void checkSeed(NumericVector seed, const std::vector<int64_t>& impliedDim)
{
  const Dimension& seedDim = seed.attr("dim");
  for (size_t i = 0; i < seedDim.size(); ++i)
  {
    if (seedDim[i] != impliedDim[i])
      throw std::runtime_error("mismatch between seed dimension " + std::to_string(i+1) + " and marginal");
  }

  // -ve seed values are not permitted
  for (size_t i = 0; i < seed.size(); ++i)
  {
    if (seed[i] < 0)
      throw std::runtime_error("negative value in seed");
  }
}

}


//' Multidimensional IPF
//'
//' C++ multidimensional IPF implementation
//' @param seed an n-dimensional array of seed values
//' @param indices a List of 1-d arrays specifying the dimension indices of each marginal as they apply to the seed values
//' @param marginals a List of arrays containing marginal data. The sum of elements in each array must be identical
//' @return an object containing:
//' \itemize{
//'   \item{a flag indicating if the solution converged}
//'   \item{the population matrix}
//'   \item{the total population}
//'   \item{the number of iterations required}
//'   \item{the maximum error between the generated population and the marginals}
//' }
//' @examples
//' ageByGender = array(c(1,2,5,3,4,3,4,5,1,2), dim=c(5,2))
//' ethnicityByGender = array(c(4,6,5,6,4,5), dim=c(3,2))
//' seed = array(rep(1,30), dim=c(5,2,3))
//' result = ipf(seed, list(c(1,2), c(3,2)), list(ageByGender, ethnicityByGender))
//' @export
// [[Rcpp::export]]
List ipf(NumericVector seed, List indices, List marginals)
{
  if (indices.size() != marginals.size())
  {
    throw std::runtime_error("index and marginal lists are different lengths");
  }

  const int64_t k = marginals.size();

  Dimension rSizes = seed.attr("dim");
  int64_t dim = rSizes.size();

  // check consistency between seed and mapped marginal dims
  Rhelpers::checkSeed(seed, Rhelpers::getDimension(indices, marginals));

  std::vector<NDArray<double>> m;
  m.reserve(k);
  std::vector<std::vector<int64_t>> idx;
  idx.reserve(k);
  std::vector<int64_t> s;
  s.reserve(dim);

  if (indices.size() != marginals.size())
    throw std::runtime_error("no. of marginals not equal to no. of indices");

  // assemble dimensions (row major) for seed
  for (int64_t i = dim-1; i >= 0; --i)
    s.push_back(rSizes[(size_t)i]);

  // insert indices and marginals in reverse order (R being column-major)
  for (int64_t i = k-1; i >= 0; --i)
  {
    const IntegerVector& iv = indices[i];
    const NumericVector& nv = marginals[i];
    idx.push_back(std::vector<int64_t>(iv.size()));
    // also need to reverse dimension indices
    for (size_t j = 0; j < iv.size(); ++j)
      idx.back()[j] = dim - iv[j];
    //std::copy(iv.begin(), iv.end(), idx.back().begin());
    // convert to NDArray
    m.push_back(std::move(Rhelpers::convertArray<double, NumericVector>(nv)));
  }

  // Storage for result
  List result;
  // Read-only shallow copy of seed
  const NDArray<double> seedwrapper(s, (double*)&seed[0]);
  // Do IPF (could provide another ctor that takes preallocated memory for result)
  IPF<double> ipf(idx, m);
  NumericVector r(rSizes);
  // Copy result data into R array
  const NDArray<double>& tmp = ipf.solve(seedwrapper);
  std::copy(tmp.rawData(), tmp.rawData() + tmp.storageSize(), r.begin());
  result["conv"] = ipf.conv();
  result["result"] = r;
  result["pop"] = ipf.population();
  result["iterations"] = ipf.iters();
  //  result["errors"] = ipf.errors();
  result["maxError"] = ipf.maxError();
  return result;
}


//' Multidimensional QIS
//'
//' C++ multidimensional Quasirandom Integer Sampling implementation
//' @param indices a List of 1-d arrays specifying the dimension indices of each marginal
//' @param marginals a List of arrays containing marginal data. The sum of elements in each array must be identical
//' @param skips (optional, default 0) number of Sobol points to skip before sampling
//' @return an object containing:
//' \itemize{
//'   \item{a flag indicating if the solution converged}
//'   \item{the population matrix}
//'   \item{the exepected state occupancy matrix}
//'   \item{the total population}
//'   \item{chi-square and p-value}
//' }
//' @examples
//' ageByGender = array(c(1,2,5,3,4,3,4,5,1,2), dim=c(5,2))
//' ethnicityByGender = array(c(4,6,5,6,4,5), dim=c(3,2))
//' result = qis(list(c(1,2), c(3,2)), list(ageByGender, ethnicityByGender))
//' @export
// [[Rcpp::export]]
List qis(List indices, List marginals, int skips = 0)
{
  if (indices.size() != marginals.size())
  {
    throw std::runtime_error("index and marginal lists are different lengths");
  }

  // we need the overall dimension and sizes upfront to assemble the problem in row-major rather than col-major form.
  std::vector<int64_t> rSizes = Rhelpers::getDimension(indices, marginals);
  // rSizes confirmed to be equivalent to dims reported by seed

  const int64_t k = marginals.size();
  const int64_t dim = rSizes.size();

  std::vector<NDArray<int64_t>> m;
  m.reserve(k);
  std::vector<std::vector<int64_t>> idx;
  idx.reserve(k);

  if (indices.size() != marginals.size())
    throw std::runtime_error("no. of marginals not equal to no. of indices");

  // insert indices and marginals in reverse order (R being column-major)
  for (int64_t i = k-1; i >= 0; --i)
  {
    const IntegerVector& iv = indices[i];
    const IntegerVector& nv = marginals[i];
    idx.push_back(std::vector<int64_t>(iv.size()));
    // also need to reverse dimension indices
    for (size_t j = 0; j < iv.size(); ++j)
      idx.back()[j] = dim - iv[j];
    // convert to NDArray
    m.push_back(std::move(Rhelpers::convertArray<int64_t, IntegerVector>(nv)));
  }

  // Storage for result
  List result;
  // Do QIS (could provide another ctor that takes preallocated memory for result)
  QIS qis(idx, m, skips);

  // How painful can it be to initialise a multidimensional array?
  int64_t size = std::accumulate(rSizes.begin(), rSizes.end(), 1ll, std::multiplies<int64_t>());
  IntegerVector r(size);
  NumericVector e(size);
  r.attr("dim") = rSizes;
  e.attr("dim") = rSizes;
  // Copy result data into R array
  const NDArray<int64_t>& tmp = qis.solve();
  std::copy(tmp.rawData(), tmp.rawData() + tmp.storageSize(), r.begin());

  const NDArray<double>& tmpe = qis.expectation();
  std::copy(tmpe.rawData(), tmpe.rawData() + tmpe.storageSize(), e.begin());
  result["conv"] = qis.conv();
  result["result"] = r;
  result["expectation"] = e;
  result["pop"] = qis.population();
  result["chiSq"] = qis.chiSq();
  result["pValue"] = qis.pValue();
  result["degeneracy"] = qis.degeneracy();

  return result;
}

//' QIS-IPF
//'
//' C++ QIS-IPF implementation
//' @param seed an n-dimensional array of seed values
//' @param indices a List of 1-d arrays specifying the dimension indices of each marginal
//' @param marginals a List of arrays containing marginal data. The sum of elements in each array must be identical
//' @param skips (optional, default 0) number of Sobol points to skip before sampling
//' @return an object containing:
//' \itemize{
//'   \item{a flag indicating if the solution converged}
//'   \item{the population matrix}
//'   \item{the exepected state occupancy matrix}
//'   \item{the total population}
//'   \item{chi-square and p-value}
//' }
//' @examples
//' ageByGender = array(c(1,2,5,3,4,3,4,5,1,2), dim=c(5,2))
//' ethnicityByGender = array(c(4,6,5,6,4,5), dim=c(3,2))
//' seed = array(rep(1,30), dim=c(5,2,3))
//' result = qisi(seed, list(c(1,2), c(3,2)), list(ageByGender, ethnicityByGender))
//' @export
// [[Rcpp::export]]
List qisi(NumericVector seed, List indices, List marginals, int skips = 0)
{
  if (indices.size() != marginals.size())
  {
    throw std::runtime_error("index and marginal lists are different lengths");
  }

  const int64_t k = marginals.size();

  Dimension rSizes = seed.attr("dim");
  const int64_t dim = rSizes.size();

  // check consistency between seed and mapped marginal dims
  Rhelpers::checkSeed(seed, Rhelpers::getDimension(indices, marginals));

  std::vector<NDArray<int64_t>> m;
  m.reserve(k);
  std::vector<std::vector<int64_t>> idx;
  idx.reserve(k);
  std::vector<int64_t> s;
  s.reserve(dim);

  if (indices.size() != marginals.size())
    throw std::runtime_error("no. of marginals not equal to no. of indices");

  // assemble dimensions (row major) for seed
  for (int64_t i = dim-1; i >= 0; --i)
    s.push_back(rSizes[(size_t)i]);

    // insert indices and marginals in reverse order (R being column-major)
  for (int64_t i = k-1; i >= 0; --i)
  {
    const IntegerVector& iv = indices[i];
    const IntegerVector& mv = marginals[i];
    idx.push_back(std::vector<int64_t>(iv.size()));
    // also need to reverse dimension indices
    for (size_t j = 0; j < iv.size(); ++j)
      idx.back()[j] = dim - iv[j];
    // convert to NDArray
    m.push_back(std::move(Rhelpers::convertArray<int64_t, IntegerVector>(mv)));
  }

  IntegerVector r(rSizes);
  NumericVector e(rSizes);

  List result;

  // Read-only shallow copy of seed
  const NDArray<double> seedwrapper(s, (double*)&seed[0]);
  // Do QIS-IPF
  QISI qisipf(idx, m, skips);

  // Copy result data into R array
  const NDArray<int64_t>& tmp = qisipf.solve(seedwrapper);
  std::copy(tmp.rawData(), tmp.rawData() + tmp.storageSize(), r.begin());
  result["result"] = r;

  // Copy result data into R array
  const NDArray<double>& tmpe = qisipf.expectation();
  std::copy(tmpe.rawData(), tmpe.rawData() + tmpe.storageSize(), e.begin());
  result["expectation"] = e;

  result["conv"] = qisipf.conv();
  result["pop"] = qisipf.population();
  result["chiSq"] = qisipf.chiSq();
  result["pValue"] = qisipf.pValue();

  return result;
}

//' Generate integer population from a fractional one where the 1-d partial sums along each axis have an integral total
//'
//' This function will generate the closest integer array to the fractional population provided, preserving the sums in every dimension.
//' @param population a numeric vector of state occupation probabilities. Must sum to unity (to within double precision epsilon)
//' @return an integer vector of frequencies that sums to pop.
//' @examples
//' prob2IntFreq(c(0.1,0.2,0.3,0.4), 11)
//' @export
// [[Rcpp::export]]
List integerise(NumericVector population)
{
  List result;
  result["conv"] = false;
  return result;
}


//' Generate integer frequencies from discrete probabilities and an overall population.
//'
//' This function will generate the closest integer vector to the probabilities scaled to the population.
//' @param pIn a numeric vector of state occupation probabilities. Must sum to unity (to within double precision epsilon)
//' @param pop the total population
//' @return an integer vector of frequencies that sum to pop, and the RMS difference from the original values.
//' @examples
//' prob2IntFreq(c(0.1,0.2,0.3,0.4), 11)
//' @export
// [[Rcpp::export]]
List prob2IntFreq(NumericVector pIn, int pop)
{
  double rmse;
  const std::vector<double>& p = as<std::vector<double>>(pIn);

  if (pop < 0)
  {
    throw std::runtime_error("population cannot be negative");
  }

  if (fabs(std::accumulate(p.begin(), p.end(), -1.0)) > 1000*std::numeric_limits<double>::epsilon())
  {
    throw std::runtime_error("probabilities do not sum to unity");
  }
  std::vector<int> f = integeriseMarginalDistribution(p, pop, rmse);

  List result;
  result["freq"] = f;
  result["rmse"] = rmse;

  return result;
}

//' Generate Sobol' quasirandom sequence
//'
//' @param dim dimensions
//' @param n number of variates to sample
//' @param skip number of variates to skip (actual number skipped will be largest power of 2 less than k)
//' @return a n-by-d matrix of uniform probabilities in (0,1).
//' @examples
//' sobolSequence(2, 1000, 1000) # will skip 512 numbers!
//' @export
// [[Rcpp::export]]
NumericMatrix sobolSequence(int dim, int n, int skip = 0)
{
  static const double scale = 0.5 / (1ull<<31);

  NumericMatrix m(n, dim);

  Sobol s(dim, skip);

  for (int j = 0; j <n ; ++j)
    for (int i = 0; i < dim; ++i)
      m(j,i) = s() * scale;

  return m;
}

//' Convert multidimensional array of counts per state into table form. Each row in the table corresponds to one individual
//'
//' This function
//' @param stateOccupancies an arbitrary-dimension array of (integer) state occupation counts.
//' @param categoryNames a string vector of unique column names.
//' @return a DataFrame with columns corresponding to category values and rows corresponding to individuals.
//' @examples
//' gender=c(51,49)
//' age=c(17,27,35,21)
//' states=qis(list(1,2),list(gender,age))$result
//' table=flatten(states,c("Gender","Age"))
//' print(nrow(table[table$Gender==1,])) # 51
//' print(nrow(table[table$Age==2,])) # 27
// [[Rcpp::export]]
DataFrame flatten(IntegerVector stateOccupancies, StringVector categoryNames)
{
  //m.push_back(std::move(convertRArray<int64_t, IntegerVector>(nv)));
  const NDArray<int64_t>& a = Rhelpers::convertArray<int64_t, IntegerVector>(stateOccupancies);
  int64_t pop = sum(a);

  // for R indices start at 1
  const std::vector<std::vector<int>>& list = listify(pop, a, 1);

  // DataFrame interface is poor and appears buggy. Best approach seems to insert columns in List then assign to DataFrame at end
  List proxyDf;
  std::string s("C");
  for (size_t i = 0; i < a.dim(); ++i)
  {
    proxyDf[as<std::string>(categoryNames[i])] = list[i];
  }

  return DataFrame(proxyDf);
}

//' Entry point to enable running unit tests within R (e.g. in testthat)
//'
//' @return a List containing, number of tests run, number of failures, and any error messages.
//' @examples
//' unitTest()
//' @export
// [[Rcpp::export]]
List unitTest()
{
  const unittest::Logger& log = unittest::run();

  List result;
  result["nTests"] = log.testsRun;
  result["nFails"] = log.testsFailed;
  result["errors"] = log.errors;

  return result;
}
