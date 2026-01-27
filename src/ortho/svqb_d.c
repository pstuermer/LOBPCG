/**
 * @file svqb_d.c
 * @brief Double precision (double) SVQB instantiation
 */

#include "../../include/lobpcg/types.h"
#include "../../lobpcg.h"
#include "../../include/lobpcg/blas_wrapper.h"
#include "../../linop.h"

#define PREFIX d
#define CTYPE f64
#define RTYPE f64
#define LINOP struct LinearOperator_d_t

#include "svqb_impl.inc"
