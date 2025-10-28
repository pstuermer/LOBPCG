#ifndef TYPES_H
#define TYPES_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <stdint.h>

typedef float f32;
typedef double f64;
typedef complex c32;
typedef double complex c64;

#define TYPE_LIST(X) \
  X(s, f32, struct LinearOperator_s_t) \
  X(d, f64, struct LinearOperator_d_t) \
  X(c, c32, struct LinearOperator_c_t) \
  X(z, c64, struct LinearOperator_z_t)

#endif
