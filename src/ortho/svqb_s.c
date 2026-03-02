/**
 * @file svqb_s.c
 * @brief Single precision (float) SVQB instantiation
 */

#include "lobpcg/types.h"
#include "lobpcg.h"
#include "lobpcg/blas_wrapper.h"
#include "lobpcg/linop.h"

#define PREFIX s
#define CTYPE f32
#define RTYPE f32
#define LINOP LinearOperator_s_t
#define TYPE_IS_FLOAT

#include "svqb_impl.inc"
