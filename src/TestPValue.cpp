
#include "UnitTester.h"

#include "PValue.h"

/*
DF=1, x = 1, r = 0.317311
DF=1, x = 2, r = 0.157299
DF=1, x = 3, r = 0.0832645
DF=1, x = 4, r = 0.0455003
DF=1, x = 5, r = 0.0253473
DF=2, x = 1, r = 0.606531
DF=2, x = 2, r = 0.367879
DF=2, x = 3, r = 0.22313
DF=2, x = 4, r = 0.135335
DF=2, x = 5, r = 0.082085
DF=3, x = 1, r = 0.801252
DF=3, x = 2, r = 0.572407
DF=3, x = 3, r = 0.391625
DF=3, x = 4, r = 0.261464
DF=3, x = 5, r = 0.171797
DF=255, x = 290.285, r = 0.0636423
*/

void unittest::testPValue()
{
  //CHECK(chiSqDensity(1, 0.0) == 0.0);
  //CHECK(chiSqDensity(2, 1.0) == 0.0);

  //LOG_INFO(format("pValue(1, 1.0) = %%", pValue(1, 1.0)));

  CHECK(withinTolerance(pValue(1, 1.0).first, 0.317311, 1e-6));
  CHECK(withinTolerance(pValue(1, 2.0).first, 0.157299, 1e-6));
  CHECK(withinTolerance(pValue(1, 3.0).first, 0.0832645, 1e-6));
  CHECK(withinTolerance(pValue(1, 4.0).first, 0.0455003, 1e-6));
  CHECK(withinTolerance(pValue(1, 5.0).first, 0.0253473, 1e-6));
  CHECK(withinTolerance(pValue(2, 1.0).first, 0.606531, 1e-6));
  CHECK(withinTolerance(pValue(2, 2.0).first, 0.367879, 1e-6));
  CHECK(withinTolerance(pValue(2, 3.0).first, 0.22313, 1e-6));
  CHECK(withinTolerance(pValue(2, 4.0).first, 0.135335, 1e-6));
  CHECK(withinTolerance(pValue(2, 5.0).first, 0.082085, 1e-6));
  CHECK(withinTolerance(pValue(3, 1.0).first, 0.801252, 1e-6));
  CHECK(withinTolerance(pValue(3, 2.0).first, 0.572407, 1e-6));
  CHECK(withinTolerance(pValue(3, 3.0).first, 0.391625, 1e-6));
  CHECK(withinTolerance(pValue(3, 4.0).first, 0.261464, 1e-6));
  CHECK(withinTolerance(pValue(3, 5.0).first, 0.171797, 1e-6));

  CHECK(withinTolerance(pValue(255, 290.285192).first, 0.0636423, 1e-6));


}
