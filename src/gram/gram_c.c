/**
 * @file gram_c.c
 * @brief Single precision complex (float complex) Gram helper instantiation
 */

#include "../../include/lobpcg/types.h"
#include "../../lobpcg.h"
#include "../../include/lobpcg/blas_wrapper.h"
#include "lobpcg/linop.h"

#define PREFIX c
#define CTYPE c32
#define RTYPE f32
#define LINOP struct LinearOperator_c_t
#define CTYPE_IS_COMPLEX

#include "gram_impl.inc"
