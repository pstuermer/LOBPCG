/**
 * @file ortho_indefinite_d.c
 * @brief Double precision (double) ortho_indefinite instantiation
 */

#include "../../include/lobpcg/types.h"
#include "../../lobpcg.h"
#include "../../include/lobpcg/blas_wrapper.h"
#include "../../linop.h"

#define PREFIX d
#define CTYPE f64
#define RTYPE f64
#define LINOP struct LinearOperator_d_t

#include "ortho_indefinite_impl.inc"
