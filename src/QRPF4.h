
#pragma once

#include "../../utils/NDArray.h"
#include "../../utils/NDArrayUtils.h"
#include "Sobol.h"


template<size_t O>
bool adjust(const std::vector<int32_t>& r, NDArray<4, uint32_t>& t)
{
  /// TODO pick index
  //std::pair<size_t,size_t> idx = pickIndex(t.size(), max(r), t);

  size_t idx[] = { 0, 0, 0, 0};

  typename NDArray<4, uint32_t>::template Iterator<O> it(t, idx);

  for(size_t i = 0; !it.end(); ++it, ++i)
  {
    *it -= r[i];
  }
}

std::vector<int32_t> diff(const std::vector<uint32_t>& x, const std::vector<uint32_t>& y)
{
  size_t size = x.size();
  assert(size == y.size());

  std::vector<int32_t> result(size);

  for (size_t i = 0; i < x.size(); ++i)
  {
    result[i] = x[i] - y[i];
  }
  return result;
}



// 4D specialisation
template<>
class QRPF<4>
{
public:

  static const size_t Dim = 4;

  typedef NDArray<4, uint32_t> table_t;

  typedef std::vector<uint32_t> marginal_t;

  QRPF(const std::vector<marginal_t>& marginals) : m_marginals(marginals)
  {
    if (m_marginals.size() != Dim)
    {
      throw std::runtime_error("invalid no. of marginals");
    }

    size_t sizes[Dim];
    m_sum = sum(m_marginals[0]);
    sizes[0] = m_marginals[0].size();
    for (size_t i = 1; i < m_marginals.size(); ++i)
    {
      if (m_sum != sum(m_marginals[i]))
      {
        throw std::runtime_error("invalid marginals");
      }
      sizes[i] = m_marginals[i].size();
    }
    m_t.resize(&sizes[0]);
  }

  ~QRPF() { }

  bool solve(size_t maxSamples = 1)
  {
    static Sobol sobol(Dim);

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

      size_t idx[Dim];
      for (size_t i = 0; i < m_sum; ++i)
      {
        for (size_t i = 0; i < Dim; ++i)
        {
          idx[i] = dists[i](sobol);
        }
        ++m_t[idx];
      }
      
      print(m_t.m_data, m_t.m_storageSize);
      
      std::vector<std::vector<int32_t>> r(Dim);
      
//      for (size_t i = 0; i < Dim; ++i)
//      {
//        r[i] = diff(reduce<Dim, uint32_t, i>(m_t), m_marginals[i]);
//      }
     
      std::cout << "initial residuals" << std::endl;
      for (size_t i = 0; i < Dim; ++i)
      {
        print(r[i]);
      }      


//      for (size_t i = 0; i < Dim; ++i)
//      {
//        if (!isZero(r[i]))
//        {
//          adjust<i>(r[i], m_t);
//          rr = diff(reduce<4, uint32_t, i>(m_t), m_marginals[i]);
//        }
//        if (!isZero(rr))
//        {
//          std::cout << "dim0 reduction failed" << std::endl;
//          print(rr);
//        }
//        
//      }

      std::cout << "adjusted residuals" << std::endl;
      for (size_t i = 0; i < Dim; ++i)
      {
        print(r[i]);
      }      

      bool success = true;
      for (size_t i = 0; i < Dim; ++i)
      {
        success = success && isZero(r[i]);
      }
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

    size_t idx[4];
    for (idx[0] = 0; idx[0] < m_t.size(0); ++idx[0])
    {
      for (idx[1] = 0; idx[1] < m_t.size(1); ++idx[1])
      {
        for (idx[2] = 0; idx[2] < m_t.size(2); ++idx[2])
        {
          for (idx[3] = 0; idx[3] < m_t.size(3); ++idx[3])
          {
            // TODO ...
            double f = double(m_marginals[0][idx[0]] * m_marginals[1][idx[1]] * m_marginals[2][idx[2]] * m_marginals[3][idx[3]]) / (m_sum*m_sum*m_sum);
            //std::cout << "f=" << f << std::endl;
            sumsq += (f - m_t[idx]) * (f - m_t[idx]);
          }
        }
      }
    }
    return sumsq / m_sum;
  }

  const table_t& result() const
  {
    return m_t;
  }

  size_t population() const
  {
    return m_sum;
  }

private:

  template<size_t O>
  void calcResiduals(std::vector<std::vector<uint32_t>>& r)
  {
    calcResiduals<O-1>(r);
    r[O-1] = diff(reduce<O, uint32_t, O-1>(m_t), m_marginals[O-1]);
  }

  const std::vector<marginal_t> m_marginals;
  table_t m_t;
  size_t m_sum;
};


template<size_t D>
template<>
void QRPF<D>::calcResiduals<1>(std::vector<std::vector<uint32_t>>& r)
{
  r[0] = diff(reduce<O, uint32_t, 0>(m_t), m_marginals[0]);  
}

