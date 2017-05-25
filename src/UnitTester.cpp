
#include "UnitTester.h"

#include <Rcpp.h>


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

  double sum = (x + y);

  if (sum < thresh)
  {
    return fabs(x - y)  < tol * sum;
  }
  return fabs(x - y)/(2 * sum) < tol;
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
  testNDArray();
  testSobol();
  testDDWR();
  testPValue();
  testQIWS();


  return Global::instance<unittest::Logger>();
}
