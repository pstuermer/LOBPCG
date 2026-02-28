/**
 * @file ortho_drop_mat_z.c
 * @brief Double precision complex (double complex) ortho_drop_mat instantiation
 */

#include "lobpcg.h"
#include "lobpcg/blas_wrapper.h"

#define PREFIX z
#define CTYPE c64
#define RTYPE f64
#define CTYPE_IS_COMPLEX

#include "ortho_drop_mat_impl.inc"
