
#include "PValue.h"
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
//
//    Output, double GAMAIN, the value of the incomplete gamma ratio.
//

double gamain ( double x, double p, int *ifault )
{
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
  double value;

  *ifault = 0;
  //
  //  Check the input.
  //
  if ( p <= 0.0 )
  {
    *ifault = 1;
    value = 0.0;
    return value;
  }

  if ( x < 0.0 )
  {
    *ifault = 2;
    value = 0.0;
    return value;
  }

  if ( x == 0.0 )
  {
    *ifault = 0;
    value = 0.0;
    return value;
  }

  g = lgamma ( p );

  arg = p * log ( x ) - x - g;

  if ( arg < loguflo )
  {
    *ifault = 3;
    value = 0.0;
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

    for ( ; ; )
    {
      rn = rn + 1.0;
      term = term * x / rn;
      gin = gin + term;

      if ( term <= acu )
      {
        break;
      }
    }

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

  for ( ; ; )
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

  return value;
}

}

double pValue(uint32_t dof, double x)
{
  double k = dof * 0.5;
  int e;
  double p = 1.0 - gamain(x/2.0, k, &e);
  if (e != 0 && e != 3) // ignore underflow errors
  {
    throw std::runtime_error("gamain error: " + std::to_string(e));
  }
  return p;
}



