/**
 * @file gram_mat_z.c
 * @brief Double precision complex (double complex) dense-matrix Gram helper instantiation
 */

#include "../../include/lobpcg/types.h"
#include "../../lobpcg.h"
#include "../../include/lobpcg/blas_wrapper.h"

#define PREFIX z
#define CTYPE c64
#define RTYPE f64
#define CTYPE_IS_COMPLEX

#include "gram_mat_impl.inc"
