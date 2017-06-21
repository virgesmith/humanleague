
#include "StatFuncs.h"
#include <limits>
#include <stdexcept>
#include <string>
#include <cmath>

namespace {

//****************************************************************************
// Adapted from: https://people.sc.fsu.edu/~jburkardt/cpp_src/asa032/asa032.html
// Licence: LGPL
//
//****************************************************************************
//
//  Purpose:
//
//    GAMAIN computes the incomplete gamma ratio.
//
//  Discussion:
//
//    A series expansion is used if P > X or X <= 1.  Otherwise, a
//    continued fraction approximation is used.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    29 June 2014
//
//  Author:
//
//    Original FORTRAN77 version by G Bhattacharjee.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    G Bhattacharjee,
//    Algorithm AS 32:
//    The Incomplete Gamma Integral,
//    Applied Statistics,
//    Volume 19, Number 3, 1970, pages 285-287.
//
//  Parameters:
//
//    Input, double X, P, the parameters of the incomplete
//    gamma ratio.  0 <= X, and 0 < P.
//
//    Output, int *IFAULT, error flag.
//    0, no errors.
//    1, P <= 0.
//    2, X < 0.
//    3, underflow.
//    4, error return from the Log Gamma routine.
//    5, failed to converge in series expansion
//    6, failed to converge in continued fraction
//
//    Output, double GAMAIN, the value of the incomplete gamma ratio.
//

double gamain ( double x, double p, int *ifault )
{
  const int maxIters = 10000;
  int iter;
  double a;
  static const double acu = 1.0E-08;
  double an;
  double arg;
  double b;
  double dif;
  double factor;
  double g;
  double gin;
  int i;
  static const double oflo = 1.0E+37;
  double pn[6];
  double rn;
  double term;
  static const double loguflo = log(1.0E-37);
  double value = 0.0;

  *ifault = 0;
  //
  //  Check the input.
  //
  if ( p <= 0.0 )
  {
    *ifault = 1;
    return value;
  }

  if ( x < 0.0 )
  {
    *ifault = 2;
    return value;
  }

  if ( x == 0.0 )
  {
    *ifault = 0;
    return value;
  }

  g = lgamma ( p );

  arg = p * log ( x ) - x - g;

  if ( arg < loguflo )
  {
    *ifault = 3;
    return value;
  }

  *ifault = 0;
  factor = exp ( arg );
  //
  //  Calculation by series expansion.
  //
  if ( x <= 1.0 || x < p )
  {
    gin = 1.0;
    term = 1.0;
    rn = p;

    for (iter = 0 ; iter < maxIters; ++iter)
    {
      rn = rn + 1.0;
      term = term * x / rn;
      gin = gin + term;

      if ( term <= acu )
      {
        break;
      }
    }
    // convergence failure
    if (iter == maxIters)
      *ifault = 5;

    value = gin * factor / p;
    return value;
  }
  //
  //  Calculation by continued fraction.
  //
  a = 1.0 - p;
  b = a + x + 1.0;
  term = 0.0;

  pn[0] = 1.0;
  pn[1] = x;
  pn[2] = x + 1.0;
  pn[3] = x * b;

  gin = pn[2] / pn[3];

  for (iter = 0 ; iter < maxIters; ++iter)
  {
    a = a + 1.0;
    b = b + 2.0;
    term = term + 1.0;
    an = a * term;
    for ( i = 0; i <= 1; i++ )
    {
      pn[i+4] = b * pn[i+2] - an * pn[i];
    }

    if ( pn[5] != 0.0 )
    {
      rn = pn[4] / pn[5];
      dif = fabs ( gin - rn );
      //
      //  Absolute error tolerance satisfied?
      //
      if ( dif <= acu )
      {
        //
        //  Relative error tolerance satisfied?
        //
        if ( dif <= acu * rn )
        {
          value = 1.0 - factor * gin;
          break;
        }
      }
      gin = rn;
    }

    for ( i = 0; i < 4; i++ )
    {
      pn[i] = pn[i+2];
    }

    if ( oflo <= fabs ( pn[4] ) )
    {
      for ( i = 0; i < 4; i++ )
      {
        pn[i] = pn[i] / oflo;
      }
    }
  }
  // convergence failure
  if (iter == maxIters)
    *ifault = 6;
  return value;
}

const double sqrt_2 = sqrt(2.0);
const double rsqrt_2 = 1.0 / sqrt(2.0);
const double sqrt_pi = sqrt(atan(1.0)*4);

// constants for Inverse Cumulative Normal approximation
const double a1_ = -3.969683028665376e+01;
const double a2_ =  2.209460984245205e+02;
const double a3_ = -2.759285104469687e+02;
const double a4_ =  1.383577518672690e+02;
const double a5_ = -3.066479806614716e+01;
const double a6_ =  2.506628277459239e+00;

const double b1_ = -5.447609879822406e+01;
const double b2_ =  1.615858368580409e+02;
const double b3_ = -1.556989798598866e+02;
const double b4_ =  6.680131188771972e+01;
const double b5_ = -1.328068155288572e+01;

const double c1_ = -7.784894002430293e-03;
const double c2_ = -3.223964580411365e-01;
const double c3_ = -2.400758277161838e+00;
const double c4_ = -2.549732539343734e+00;
const double c5_ =  4.374664141464968e+00;
const double c6_ =  2.938163982698783e+00;

const double d1_ =  7.784695709041462e-03;
const double d2_ =  3.224671290700398e-01;
const double d3_ =  2.445134137142996e+00;
const double d4_ =  3.754408661907416e+00;

// Limits of the approximation regions
const double x_low_ = 0.02425;
const double x_high_= 1.0 - x_low_;

// tail approximation
double tail_value(double x)
{
  if (x <= 0.0 || x >= 1.0)
  {
    // try to recover if due to numerical error
    if (std::fabs(x - 1.0) <  4 * std::numeric_limits<double>::epsilon())
    {
      return std::numeric_limits<double>::max(); // largest value available
    }
    else if (std::fabs(x) < std::numeric_limits<double>::epsilon())
    {
      return -std::numeric_limits<double>::max(); // largest negative value available
    }
    else
    {
      throw std::runtime_error("Inverse cumulative normal: x must be in [0,1]");
    }
  }

  double z;
  if (x < x_low_)
  {
    // Rational approximation for the lower region 0<x<u_low
    z = std::sqrt(-2.0*std::log(x));
    z = (((((c1_*z+c2_)*z+c3_)*z+c4_)*z+c5_)*z+c6_) /
      ((((d1_*z+d2_)*z+d3_)*z+d4_)*z+1.0);
  }
  else
  {
    // Rational approximation for the upper region u_high<x<1
    z = std::sqrt(-2.0*std::log(1.0-x));
    z = -(((((c1_*z+c2_)*z+c3_)*z+c4_)*z+c5_)*z+c6_) /
      ((((d1_*z+d2_)*z+d3_)*z+d4_)*z+1.0);
  }

  return z;
}

}


double cumNorm(double x)
{
  return 0.5*(erf(x * rsqrt_2) + 1.0);
}


double invCumNorm(const double x)
{
  double z;
  if (x < x_low_ || x_high_ < x) {
    z = tail_value(x);
  } else {
    z = x - 0.5;
    double r = z*z;
    z = (((((a1_*r+a2_)*r+a3_)*r+a4_)*r+a5_)*r+a6_)*z /
      (((((b1_*r+b2_)*r+b3_)*r+b4_)*r+b5_)*r+1.0);
  }

  // The relative error of the approximation has absolute value less
  // than 1.15e-9.  One iteration of Halley's rational method (third
  // order) gives full machine precision.

  // error (cumNorm(z) - x) divided by the cumulative's derivative
  const double r = (cumNorm(z) - x) * sqrt_2 * sqrt_pi * exp(0.5 * z*z);
  //  Halley's method
  z -= r/(1+0.5*z*r);

  return z;
}


// Cholesky factorisation for single correlation (2x2 matrix)
std::array<double, 4> cholesky(double rho)
{
  if (std::fabs(rho) > 1.0)
    throw std::runtime_error("correlation is not in [-1,1]");
  std::array<double,4> ret;
  ret[0] = 1.0;
  ret[1] = 0.0;
  ret[2] = rho;
  ret[3] = sqrt(1.0 - rho * rho);
  return ret;
}


std::pair<double, bool> pValue(uint32_t dof, double x)
{
  double k = dof * 0.5;
  int e = 0;
  double p = 1.0 - gamain(x/2.0, k, &e);
  // we drop detail of the error here and just return true/false, plus the value (false meaning p is potentially inaccurate)
  return std::make_pair(p, e == 0);
}



