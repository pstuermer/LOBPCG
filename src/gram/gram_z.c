/**
 * @file gram_z.c
 * @brief Double precision complex (double complex) Gram helper instantiation
 */

#include "../../include/lobpcg/types.h"
#include "../../lobpcg.h"
#include "../../include/lobpcg/blas_wrapper.h"
#include "lobpcg/linop.h"

#define PREFIX z
#define CTYPE c64
#define RTYPE f64
#define LINOP struct LinearOperator_z_t
#define CTYPE_IS_COMPLEX

#include "gram_impl.inc"
