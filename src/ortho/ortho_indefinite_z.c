/**
 * @file ortho_indefinite_z.c
 * @brief Double precision complex (double complex) ortho_indefinite instantiation
 */

#include "lobpcg.h"
#include "lobpcg/linop.h"
#include "lobpcg/blas_wrapper.h"

#define PREFIX z
#define CTYPE c64
#define RTYPE f64
#define LINOP LinearOperator_z_t
#define CTYPE_IS_COMPLEX

#include "ortho_indefinite_impl.inc"
