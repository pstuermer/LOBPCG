/**
 * @file svqb_d.c
 * @brief Double precision (double) SVQB instantiation
 */

#include "lobpcg/types.h"
#include "lobpcg.h"
#include "lobpcg/blas_wrapper.h"
#include "lobpcg/linop.h"

#define PREFIX d
#define CTYPE f64
#define RTYPE f64
#define LINOP LinearOperator_d_t

#include "svqb_impl.inc"
