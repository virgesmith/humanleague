
#pragma once

#include "QIWS.h"


// 2-Dimensional constrained quasirandom integer without-replacement sampling
// constraint is hard-coded (for now) to: idx1 <= idx0
// TODO rename
//template<size_t D>
class RQIWS : public QIWS<2>
{
public:

  RQIWS(const std::vector<marginal_t>& marginals, double rho);

  ~RQIWS() { }

  bool solve();

private:
  const double m_rho;
};





