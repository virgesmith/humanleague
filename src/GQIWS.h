
#pragma once

#include "QIWS.h"

// struct Constrain
// {
//   enum Status { SUCCESS = 0, ITERLIMIT = 1, STUCK = 2 };
// };


// 2-Dimensional constrained quasirandom integer without-replacement sampling
// constraint is hard-coded (for now) to: idx1 <= idx0
// TODO rename
//template<size_t D>
class GQIWS : public QIWS<2>
{
public:

  GQIWS(const std::vector<marginal_t>& marginals, const NDArray<2, double>& exoProbs);

  ~GQIWS() { }

  bool solve();

  // allow constraining of precomputed populations
  //static Constrain::Status constrain(NDArray<2, uint32_t>& pop, const NDArray<2, bool>& allowedStates, const size_t iterLimit);

private:
  // no copy semantics so just store a ref (possibly dangerous)
  const NDArray<2, double>& m_exoprobs;

};





