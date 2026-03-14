#ifndef TYPE_DISPATCH_H
#define TYPE_DISPATCH_H

#include <math.h>
#include <complex.h>

#ifdef TYPE_IS_FLOAT
  #define RABS(x) fabsf(x)
  #define RSQRT(x) sqrtf(x)
  #ifdef CTYPE_IS_COMPLEX
    #define CABS(x) cabsf(x)
    #define CREAL(x) creal(x)
  #else
    #define CABS(x) fabsf(x)
    #define CREAL(x) (x)
  #endif
#else
  #define RABS(x) fabs(x)
  #define RSQRT(x) sqrt(x)
  #ifdef CTYPE_IS_COMPLEX
    #define CABS(x) cabs(x)
    #define CREAL(x) creal(x)
  #else
    #define CABS(x) fabs(x)
    #define CREAL(x) (x)
  #endif
#endif

#endif // TYPE_DISPATCH_H
