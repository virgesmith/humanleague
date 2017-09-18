// UnitTester.h

#include "Global.h"

#include <vector>
#include <string>

namespace unittest {

struct Logger
{
  Logger() : testsRun(0), testsFailed(0) { }

  // required since object will persist state as long as library is loaded
  void reset();

  size_t testsRun;
  size_t testsFailed;
  std::vector<std::string> errors;
};

bool withinTolerance(double x, double y, double tol = std::numeric_limits<double>::epsilon());

}

// Test macros
#define CHECK(cond)                                                                         \
 ++Global::instance<unittest::Logger>().testsRun;                                           \
 if (!(cond))                                                                               \
 {                                                                                          \
   ++Global::instance<unittest::Logger>().testsFailed;                                      \
   Global::instance<unittest::Logger>().errors.push_back(std::string(#cond) + " FAILED at " \
                              + __FILE__ + ":" + std::to_string(__LINE__));                 \
 }

#define CHECK_THROWS(expr, except)                                             \
  {                                                                            \
    ++Global::instance<unittest::Logger>().testsRun;                           \
    bool caught = false;                                                       \
    try                                                                        \
    {                                                                          \
      expr;                                                                    \
    }                                                                          \
    catch(except& e)                                                           \
    {                                                                          \
      caught = true;                                                           \
    }                                                                          \
    catch(...)                                                                 \
    {                                                                          \
    }                                                                          \
    if (!caught)                                                               \
    {                                                                          \
      ++Global::instance<unittest::Logger>().testsFailed;                      \
      Global::instance<unittest::Logger>().errors.push_back(std::string(#expr) + " did not throw expected " #except \
      + " at " __FILE__ + ":" + std::to_string(__LINE__));                     \
    }                                                                          \
  }

#define UNEXPECTED_ERROR(msg) \
  ++Global::instance<unittest::Logger>().testsRun;    \
  ++Global::instance<unittest::Logger>().testsFailed; \
  Global::instance<unittest::Logger>().errors.push_back(std::string("Unexpected error ") +  msg + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \


#define UNHANDLED_ERROR()     \
  ++Global::instance<unittest::Logger>().testsRun;    \
  ++Global::instance<unittest::Logger>().testsFailed; \
  Global::instance<unittest::Logger>().errors.push_back(std::string("Unhandled error at ") + __FILE__ + ":" + std::to_string(__LINE__)); \

namespace unittest {

// insert test function declarations here
void testNDArray();
void testSobol();
void testDDWR();
void testCumNorm();
void testCholesky();
void testPValue();
void testQIWS();
//void testConstrainedSampling();

const Logger& run();

}
