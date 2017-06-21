
#include "UnitTester.h"

#include "StatFuncs.h"

#include <cmath>

#include <iostream>
#include <iomanip>

// TODO add unit tests for cholesky and (inv) cum norm

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


// tests cumulative normal and its inverse
void unittest::testCumNorm()
{
  CHECK(cumNorm(-1e100) == 0.0);
  CHECK(cumNorm(-8.2) < std::numeric_limits<double>::epsilon());
  CHECK(cumNorm(0.0) == 0.5);
  CHECK(cumNorm(+8.2) >  1.0 - std::numeric_limits<double>::epsilon());
  CHECK(cumNorm(+1e100) == 1.0);

  //CHECK(std::isinf(invCumNorm(0.0)));
  //std::cout << invCumNorm(0.0) << std::endl;
  CHECK(invCumNorm(0.5) == 0.0);
  //CHECK(std::isinf(invCumNorm(1.0)));

  for (double x = -6.05; x <= 6.0; x += 0.1)
  {
    CHECK(withinTolerance(x, invCumNorm(cumNorm(x)), 0.001));
  }
  for (double x = 0.01; x <= 0.99; x += 0.01)
  {
    CHECK(withinTolerance(x, cumNorm(invCumNorm(x)), std::numeric_limits<double>::epsilon() * 10));
  }
}

void unittest::testCholesky()
{
  double rho = 0.0;

  auto func = [](double rho){
    std::array<double, 4> m = cholesky(rho);
    CHECK(m[0] == 1.0);
    CHECK(m[1] == 0.0);
    CHECK(m[2] == rho);
    CHECK(m[3] == sqrt(1.0-rho*rho));
  };
  // std::array<double, 4> m = cholesky(rho);
  // CHECK(m[0] == 1.0);
  // CHECK(m[1] == 0.0);
  // CHECK(m[2] == rho);
  // CHECK(m[3] == sqrt(1.0-rho*rho));
  func(-1.0);
  func(-0.5);
  func(0.0);
  func(0.5);
  func(1.0);
  CHECK_THROWS(func(2.0), std::exception);

}

void unittest::testPValue()
{
  //CHECK(chiSqDensity(1, 0.0) == 0.0);
  //CHECK(chiSqDensity(2, 1.0) == 0.0);

  //LOG_INFO(format("pValue(1, 1.0) = %%", pValue(1, 1.0)));

  CHECK(withinTolerance(pValue(1, 1.0).first, 0.317311, 2e-6));
  CHECK(withinTolerance(pValue(1, 2.0).first, 0.157299, 2e-6));
  CHECK(withinTolerance(pValue(1, 3.0).first, 0.0832645, 1e-6));
  CHECK(withinTolerance(pValue(1, 4.0).first, 0.0455003, 1e-6));
  CHECK(withinTolerance(pValue(1, 5.0).first, 0.0253473, 1e-6));
  CHECK(withinTolerance(pValue(2, 1.0).first, 0.606531, 1e-6));
  CHECK(withinTolerance(pValue(2, 2.0).first, 0.367879, 2e-6));
  CHECK(withinTolerance(pValue(2, 3.0).first, 0.22313, 1e-6));
  CHECK(withinTolerance(pValue(2, 4.0).first, 0.135335, 4e-6));
  CHECK(withinTolerance(pValue(2, 5.0).first, 0.082085, 1e-6));
  CHECK(withinTolerance(pValue(3, 1.0).first, 0.801252, 1e-6));
  CHECK(withinTolerance(pValue(3, 2.0).first, 0.572407, 1e-6));
  CHECK(withinTolerance(pValue(3, 3.0).first, 0.391625, 1e-6));
  CHECK(withinTolerance(pValue(3, 4.0).first, 0.261464, 1e-6));
  CHECK(withinTolerance(pValue(3, 5.0).first, 0.171797, 1e-6));

  CHECK(withinTolerance(pValue(255, 290.285192).first, 0.0636423, 1e-6));
}
