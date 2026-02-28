/**
 * @file gram_mat_c.c
 * @brief Single precision complex (float complex) dense-matrix Gram helper instantiation
 */

#include "../../include/lobpcg/types.h"
#include "../../lobpcg.h"
#include "../../include/lobpcg/blas_wrapper.h"

#define PREFIX c
#define CTYPE c32
#define RTYPE f32
#define CTYPE_IS_COMPLEX

#include "gram_mat_impl.inc"
