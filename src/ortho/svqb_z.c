/**
 * @file svqb_z.c
 * @brief Double precision complex (double complex) SVQB instantiation
 */

#include "../../include/lobpcg/types.h"
#include "../../lobpcg.h"
#include "../../include/lobpcg/blas_wrapper.h"
#include "../../linop.h"

#define PREFIX z
#define CTYPE c64
#define RTYPE f64
#define LINOP struct LinearOperator_z_t
#define CTYPE_IS_COMPLEX

#include "svqb_impl.inc"
