
#include "UnitTester.h"

//#include <Rcpp.h>
#include <cmath>

void unittest::Logger::reset()
{
  testsRun = 0;
  testsFailed = 0;
  errors.clear();
}

bool unittest::withinTolerance(double x, double y, double tol)
{
  // this check is symmetric and relative, unless there is an overflow/loss of precision then its absolute
  static const double thresh = std::numeric_limits<double>::min() / std::numeric_limits<double>::epsilon();

  double mean = 0.5 * fabs(x + y);

  //std::cout << x << ", " << y << ", " << mean << ", " << fabs(x-y) << ", " << tol << ", " << thresh << std::endl;

  if (mean < thresh)
  {
    return fabs(x - y)  < tol * mean;
  }

  return fabs(x - y) / mean < tol;
}

const unittest::Logger& unittest::run()
{
  // Reset test logger state
  Global::instance<Logger>().reset();

  // Example failures
  // CHECK(1==0);
  // CHECK_THROWS(2+2, std::runtime_error);
  // UNEXPECTED_ERROR("Testing unexpected");
  // UNHANDLED_ERROR();

  // call test functions
  //testConstrainedSampling();
  testNDArray();
  testSobol();
  //testDDWR();
  testCumNorm();
  testCholesky();
  testPValue();
  testQIWS();

  testIndex();
  testSlice();
  testReduce();

  return Global::instance<Logger>();
}
