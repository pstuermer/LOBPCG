/**
 * @file gram_d.c
 * @brief Double precision (double) Gram helper instantiation
 */

#include "../../include/lobpcg/types.h"
#include "../../lobpcg.h"
#include "../../include/lobpcg/blas_wrapper.h"
#include "lobpcg/linop.h"

#define PREFIX d
#define CTYPE f64
#define RTYPE f64
#define LINOP struct LinearOperator_d_t

#include "gram_impl.inc"
