/**
 * @file ortho_indefinite_mat_c.c
 * @brief Single precision complex (float complex) ortho_indefinite_mat instantiation
 */

#include "lobpcg.h"
#include "lobpcg/blas_wrapper.h"

#define PREFIX c
#define CTYPE c32
#define RTYPE f32
#define CTYPE_IS_COMPLEX
#define TYPE_IS_FLOAT

#include "ortho_indefinite_mat_impl.inc"
