
#pragma once

#include "QIWS.h"

struct Constrain
{
  enum Status { SUCCESS = 0, ITERLIMIT = 1, STUCK = 2 };
};


// 2-Dimensional constrained quasirandom integer without-replacement sampling
// constraint is hard-coded (for now) to: idx1 <= idx0
// TODO rename
//template<size_t D>
class CQIWS : public QIWS<2>
{
public:

  CQIWS(const std::vector<marginal_t>& marginals, const NDArray<2, bool>& permittedStates);

  ~CQIWS() { }

  bool solve();

private:
  // no copy semantics so just store a ref (possibly dangerous)
  const NDArray<2, bool>& m_allowedStates;

};





