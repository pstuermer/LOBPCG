/**
 * @file ortho_drop_mat_s.c
 * @brief Single precision (float) ortho_drop_mat instantiation
 */

#include "lobpcg.h"
#include "lobpcg/blas_wrapper.h"

#define PREFIX s
#define CTYPE f32
#define RTYPE f32

#include "ortho_drop_mat_impl.inc"
