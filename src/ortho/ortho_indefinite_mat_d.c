/**
 * @file ortho_indefinite_mat_d.c
 * @brief Double precision (double) ortho_indefinite_mat instantiation
 */

#include "lobpcg.h"
#include "lobpcg/blas_wrapper.h"

#define PREFIX d
#define CTYPE f64
#define RTYPE f64

#include "ortho_indefinite_mat_impl.inc"
