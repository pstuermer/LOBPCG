/**
 * @file ortho_indefinite_mat_s.c
 * @brief Single precision (float) ortho_indefinite_mat instantiation
 */

#include "lobpcg.h"
#include "lobpcg/blas_wrapper.h"

#define PREFIX s
#define CTYPE f32
#define RTYPE f32
#define TYPE_IS_FLOAT

#include "ortho_indefinite_mat_impl.inc"
