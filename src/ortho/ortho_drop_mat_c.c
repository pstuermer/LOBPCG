/**
 * @file ortho_drop_mat_c.c
 * @brief Single precision complex (float complex) ortho_drop_mat instantiation
 */

#include "lobpcg.h"
#include "lobpcg/blas_wrapper.h"

#define PREFIX c
#define CTYPE c32
#define RTYPE f32
#define CTYPE_IS_COMPLEX

#include "ortho_drop_mat_impl.inc"
