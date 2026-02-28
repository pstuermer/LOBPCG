/**
 * @file gram_mat_d.c
 * @brief Double precision (double) dense-matrix Gram helper instantiation
 */

#include "../../include/lobpcg/types.h"
#include "../../lobpcg.h"
#include "../../include/lobpcg/blas_wrapper.h"

#define PREFIX d
#define CTYPE f64
#define RTYPE f64

#include "gram_mat_impl.inc"
