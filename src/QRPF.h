#pragma once
#include "Sobol.h"
#include "Tensor.h"

#include <vector>
#include <random>

#include <iostream>

template<typename T>
void print(const Tensor<1, T>& v)
{
  for (size_t i = 0; i < v.size(); ++i)
  {
    std::cout << v[i] << " ";
  }
  std::cout << ";" << std::endl;

}

template<size_t D, typename T>
void print(const Tensor<D, T>& t)
{
  for (size_t i = 0; i < t.size(); ++i)
  {
    print(t[i]);
  }
}


// TODO generalise to N-D
Tensor<1, int32_t> diff(const Tensor<1, uint32_t>& x, const Tensor<1, uint32_t>& y)
{
  size_t size = x.size();
  Tensor<1, int32_t> result(&size);
  if (x.size() != y.size())
    throw std::runtime_error("size mismatch");
  for (size_t i = 0; i < x.size(); ++i)
  {
    result[i] = x[i] - y[i];
  }
  return result;
}

Tensor<2, int32_t> diff(const Tensor<2, uint32_t>& x, const Tensor<2, uint32_t>& y)
{
  size_t sizes[2] = { x.size(), x[0].size() };
  Tensor<2, int32_t> result(sizes);
  if (x.size() != y.size() || x[0].size() != y[0].size())
    throw std::runtime_error("size mismatch");

  for (size_t j = 0; j < x[0].size(); ++j)
  {
    for (size_t i = 0; i < x.size(); ++i)
    {
      result[i][j] = x[i][j] - y[i][j];
    }
  }
  return result;
}


// picks a random index that won't cause a -ve value in t
// TODO generalise
size_t pickIndex(size_t size, int32_t maxVal, const Tensor<2, uint32_t>& t)
{
  // get a list of indices where min value is large enough
  std::vector<size_t> iValid;
  for (size_t i = 0; i < size; ++i)
  {
    if (static_cast<int32_t>(min(t[i])) >= maxVal)
      iValid.push_back(i);
//    else
//    {
//      std::cout << "index " << i << " " << min(t[i]) << " < " << maxVal << std::endl;
//    }
  }

  // select one at random
  static std::mt19937 mt(7237);
  std::uniform_int_distribution<uint32_t> d(0, iValid.size()-1);

  if (iValid.empty())
  {
    return -1;
  }
  else
  {
    return iValid[d(mt)];
  }
}

// picks a random index such that subtracting maxVal won't result in a -ve value in t
// TODO generalise
std::pair<size_t, size_t> pickIndex(size_t size, int32_t maxVal, const Tensor<3, uint32_t>& t)
{
  // get a list of indices where min value is large enough
  std::vector<size_t> iValid;
  std::vector<size_t> jValid;
  for (size_t i = 0; i < t.size(); ++i)
  {
    for (size_t j = 0; j < t[0].size(); ++j)
    {
      if (static_cast<int32_t>(min(t[i][j])) >= maxVal)
      {
        iValid.push_back(i);
        jValid.push_back(j);
      }
    }
  }

  // select one at random
  static std::mt19937 mt(7237);
  std::uniform_int_distribution<uint32_t> di(0, iValid.size()-1);
  std::uniform_int_distribution<uint32_t> dj(0, jValid.size()-1);

  if (iValid.empty() || jValid.empty())
  {
    return std::make_pair(-1ull, -1ull);
  }
  else
  {
    return std::make_pair(iValid[di(mt)], jValid[dj(mt)]);
  }
}


// picks a random index that won't cause a -ve value in t
// TODO generalise
size_t pickIndex2(size_t size, int32_t maxVal, const Tensor<2, uint32_t>& t)
{
  // get a list of indices where min value is large enough
  std::vector<size_t> iValid;
  for (size_t i = 0; i < size; ++i)
  {
    if (static_cast<int32_t>(min(t,i)) >= maxVal)
      iValid.push_back(i);
  }

  // select one at random
  static std::mt19937 mt(7237);
  std::uniform_int_distribution<uint32_t> d(0, iValid.size()-1);

  if (iValid.empty())
  {
    return -1;
  }
  else
  {
    return iValid[d(mt)];
  }
}

bool adjust(const Tensor<1, int32_t>& r, size_t d, Tensor<2, uint32_t>& t)
{
  if (d == 1)
  {
    // pick an index
    size_t idx = pickIndex(t.size(), max(r), t);
    if (idx == -1ull)
      return false;
    //std::cout << "idx=" << idx << std::endl;

    for (size_t i = 0; i < r.size(); ++i)
    {
      t[idx][i] -= r[i];
    }
  }
  else
  {
    // pick an index
    size_t idx = pickIndex2(t[0].size(), max(r), t);
    if (idx == -1ull)
      return false;

    //std::cout << "idx=" << idx << std::endl;

    for (size_t i = 0; i < r.size(); ++i)
    {
      t[i][idx] -= r[i];
    }
  }
  return true;
}

bool adjust(const Tensor<1, int32_t>& r, size_t d, Tensor<3, uint32_t>& t)
{
  if (d == 2)
  {
    std::pair<size_t,size_t> idx = pickIndex(t.size(), max(r), t);
    //std::cout << d << "->" << idx.first << "," << idx.second << std::endl;
    if (idx.first == -1ull || idx.second == -1ull)
      return false;

    for (size_t i = 0; i < r.size(); ++i)
    {
      t[idx.first][idx.second][i] -= r[i];
    }
  }
  else if (d == 1)
  {
    std::pair<size_t, size_t> idx = std::make_pair(t.size()/2, t[0][0].size()/2);
    //std::pair<size_t,size_t> idx = pickIndex(t.size(), max(r), t);
    //std::cout << d << "->" << idx.first << "," << idx.second << std::endl;
    for (size_t i = 0; i < r.size(); ++i)
    {
      t[idx.first][i][idx.second] -= r[i];
    }
  }
  else
  {
    std::pair<size_t, size_t> idx = std::make_pair(t[0].size()/2, t[0][0].size()/2);
    //std::pair<size_t,size_t> idx = pickIndex(t.size(), max(r), t);
    //std::cout << d << "->" << idx.first << "," << idx.second << std::endl;
    for (size_t i = 0; i < r.size(); ++i)
    {
      t[i][idx.first][idx.second] -= r[i];
    }
  }
  return true;
}

size_t worstCase = 0;

// Quasirandom proportional fitting
template<size_t D>
class QRPF
{
public:

  static const size_t Dim = D;

  typedef Tensor<D, uint32_t> table_t;

  // TODO are marginals 1-d or (D-1)-d ??? do we construct (D-1)-d marginals by recursively applying QRPF<D-1>?
  // assuming D-1 (possibly constructed from D-2...)
  typedef Tensor<D-1, uint32_t> marginal_t;

  QRPF(const std::vector<marginal_t>& marginals) : m_marginals(marginals)/*, m_sobol(D)*/
  {
    if (m_marginals.size() != Dim)
    {
      throw std::runtime_error("invalid no. of marginals");
    }

    std::vector<size_t> sizes(Dim);
    m_sum = m_marginals[0].sum();
    // TODO ordering other way?
    sizes[Dim-1] = m_marginals[0].size();
    for (size_t i = 1; i < m_marginals.size(); ++i)
    {
      if (m_sum != m_marginals[i].sum())
      {
        throw std::runtime_error("invalid marginals");
      }
      sizes[Dim-i-1] = m_marginals[i].size();
    }
    m_t.resize(&sizes[0]);
  }

  ~QRPF() { }

  bool solve(size_t maxSamples = 4)
  {
    static Sobol sobol(2);

    std::vector<std::discrete_distribution<uint32_t>> dists;
    for (size_t i = 0; i < Dim; ++i)
    {
      dists.push_back(std::discrete_distribution<uint32_t>(m_marginals[i].begin(), m_marginals[i].end()));
    }

    size_t sample = 0;
    bool success = false;

    for (; sample < maxSamples; ++sample)
    {
      m_t.assign(0u);
      for (size_t i = 0; i < m_sum; ++i)
      {
        // TODO reverse order is suboptimal
        std::vector<uint32_t> idx(Dim);
        for (size_t i = 0; i < Dim; ++i)
        {
          idx[Dim - i - 1] = dists[i](sobol);
        }
        ++m_t.at(&idx[0]);
      }

      Tensor<1, int32_t> rr = diff(reduce(0, m_t), m_marginals[1]);
      Tensor<1, int32_t> cr = diff(reduce(1, m_t), m_marginals[0]);

      if (!isZero(rr))
      {
        if (!adjust(rr, 0, m_t))
        {
          //std::cout << "initial guess invalid (r)" << std::endl;
          continue;
        }
        rr = diff(reduce(0, m_t), m_marginals[1]);
      }

      if (!isZero(cr))
      {
        if (!adjust(cr, 1, m_t))
        {
          //std::cout << "initial guess invalid (c)" << std::endl;
          continue;
        }
        cr = diff(reduce(1, m_t), m_marginals[0]);
      }

      success = isZero(rr) && isZero(cr);

      if (success)
        break;

//      std::cout << "final residuals" << std::endl;
//      print(rr);
//      std::cout << isZero(rr) << std::endl;
//      print(cr);
//      std::cout << isZero(rr) << std::endl;
//      print(m_t);
    }
    if (success) worstCase = std::max(worstCase, sample);


    return success;
  }

  const table_t& result() const
  {
    return m_t;
  }

  double msv() const
  {
    double sumsq = 0.0;

    for (size_t i = 0; i < m_t.size(); ++i)
    {
      for (size_t j = 0; j <m_t[0].size(); ++j)
      {
        double f = double(m_marginals[1][i] * m_marginals[0][j]) / m_sum;
        sumsq += (f - m_t[i][j]) * (f - m_t[i][j]);
      }
    }
    return sumsq / m_sum;
  }

  uint32_t sum() const
  {
    return m_sum;
  }


private:

  const std::vector<marginal_t> m_marginals;
  table_t m_t;
  uint32_t m_sum;
  //uint32_t m_iterations;
  //Sobol m_sobol;
};


// 3D specialisation
template<>
class QRPF<3>
{
public:

  static const size_t Dim = 3;

  typedef Tensor<3, uint32_t> table_t;

  // TODO are marginals 1-d or (D-1)-d ??? do we construct (D-1)-d marginals by recursively applying QRPF<D-1>?
  // assuming D-1 (possibly constructed from D-2...)
  typedef Tensor<1, uint32_t> marginal_t;

  QRPF(const std::vector<marginal_t>& marginals) : m_marginals(marginals)/*, m_sobol(3)*/
  {
    if (m_marginals.size() != Dim)
    {
      throw std::runtime_error("invalid no. of marginals");
    }

    std::vector<size_t> sizes(Dim);
    m_sum = m_marginals[0].sum();
    // TODO ordering other way?
    sizes[Dim-1] = m_marginals[0].size();
    for (size_t i = 1; i < m_marginals.size(); ++i)
    {
      if (m_sum != m_marginals[i].sum())
      {
        throw std::runtime_error("invalid marginals");
      }
      sizes[Dim-i-1] = m_marginals[i].size();
    }
    m_t.resize(&sizes[0]);
  }

  ~QRPF() { }

  bool solve(size_t maxSamples = 1)
  {
    static Sobol sobol(3);

    std::vector<std::discrete_distribution<uint32_t>> dists;
    for (size_t i = 0; i < Dim; ++i)
    {
      dists.push_back(std::discrete_distribution<uint32_t>(m_marginals[i].begin(), m_marginals[i].end()));
    }

    size_t sample = 0;
    bool success = false;

    for (; sample < maxSamples && !success; ++sample)
    {
      m_t.assign(0u);

      for (size_t i = 0; i < m_sum; ++i)
      {
        // TODO reverse order is suboptimal
        std::vector<uint32_t> idx(Dim);
        for (size_t i = 0; i < Dim; ++i)
        {
          idx[Dim - i - 1] = dists[i](sobol);
        }
        ++m_t.at(&idx[0]);
      }

      Tensor<1, int32_t> rr = diff(reduce(0, m_t), m_marginals[2]);
      Tensor<1, int32_t> cr = diff(reduce(1, m_t), m_marginals[1]);
      Tensor<1, int32_t> sr = diff(reduce(2, m_t), m_marginals[0]);
//      std::cout << "initial residuals" << std::endl;
//      print(rr);
//      print(cr);
//      print(sr);

      if (!isZero(rr))
      {
        adjust(rr, 0, m_t);
        rr = diff(reduce(0, m_t), m_marginals[2]);
      }
//      if (!isZero(rr))
//      {
//        std::cout << "row reduction failed" << std::endl;
//        print(rr);
//      }
      if (!isZero(cr))
      {
        adjust(cr, 1, m_t);
        cr = diff(reduce(1, m_t), m_marginals[1]);
      }
//      if (!isZero(cr))
//      {
//        std::cout << "col reduction failed" << std::endl;
//        print(cr);
//      }
      if (!isZero(sr))
      {
        adjust(sr, 2, m_t);
        sr = diff(reduce(2, m_t), m_marginals[0]);
      }
//      if (!isZero(sr))
//      {
//        std::cout << "slice reduction failed" << std::endl;
//        print(sr);
//      }

//      std::cout << "adjusted residuals" << std::endl;
//      print(rr);
//      print(cr);
//      print(sr);
      success = isZero(rr) && isZero(cr) && isZero(sr);

      // temporary check we dont have a -ve value (until index picking is fixed)
      success = success && (max(m_t) < 1ull<<31);
    }

    //print(m_t);
    return success;
  }

  double msv() const
  {
    double sumsq = 0.0;

    /*std::cout << "m_t.size()" << m_t.size()
              << "\nm_t[0].size()" << m_t[0].size()
              << "\nm_t[0][0].size()" << m_t[0][0].size() << std::endl;
    std::cout << "m_marginals[0]" << m_marginals[0].size()
              << "\nm_marginals[1]" << m_marginals[1].size()
              << "\nm_marginals[2]" << m_marginals[2].size() << std::endl;*/

    for (size_t i = 0; i < m_t.size(); ++i)
    {
      for (size_t j = 0; j < m_t[0].size(); ++j)
      {
        for (size_t k = 0; k < m_t[0][0].size(); ++k)
        {
          double f = double(m_marginals[1][j] * m_marginals[0][k] * m_marginals[2][i]) / (m_sum*m_sum);
          //std::cout << "f=" << f << std::endl;
          sumsq += (f - m_t[i][j][k]) * (f - m_t[i][j][k]);
        }
      }
    }
    return sumsq / m_sum;
  }

  const table_t& result() const
  {
    return m_t;
  }

  uint32_t sum() const
  {
    return m_sum;
  }

private:

  const std::vector<marginal_t> m_marginals;
  table_t m_t;
  uint32_t m_sum;
};



