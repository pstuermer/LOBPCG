/**
 * @file ortho_indefinite_d.c
 * @brief Double precision (double) ortho_indefinite instantiation
 */

#include "lobpcg.h"
#include "lobpcg/linop.h"
#include "lobpcg/blas_wrapper.h"

#define PREFIX d
#define CTYPE f64
#define RTYPE f64
#define LINOP LinearOperator_d_t

#include "ortho_indefinite_impl.inc"
