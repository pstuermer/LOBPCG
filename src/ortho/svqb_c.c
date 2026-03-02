/**
 * @file svqb_c.c
 * @brief Single precision complex (float complex) SVQB instantiation
 */

#include "lobpcg/types.h"
#include "lobpcg.h"
#include "lobpcg/blas_wrapper.h"
#include "lobpcg/linop.h"

#define PREFIX c
#define CTYPE c32
#define RTYPE f32
#define LINOP LinearOperator_c_t
#define CTYPE_IS_COMPLEX
#define TYPE_IS_FLOAT

#include "svqb_impl.inc"
