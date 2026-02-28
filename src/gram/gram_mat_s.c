/**
 * @file gram_mat_s.c
 * @brief Single precision (float) dense-matrix Gram helper instantiation
 */

#include "../../include/lobpcg/types.h"
#include "../../lobpcg.h"
#include "../../include/lobpcg/blas_wrapper.h"

#define PREFIX s
#define CTYPE f32
#define RTYPE f32

#include "gram_mat_impl.inc"
