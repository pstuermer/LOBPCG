/**
 * @file gram_s.c
 * @brief Single precision (float) Gram helper instantiation
 */

#include "../../include/lobpcg/types.h"
#include "../../lobpcg.h"
#include "../../include/lobpcg/blas_wrapper.h"
#include "lobpcg/linop.h"

#define PREFIX s
#define CTYPE f32
#define RTYPE f32
#define LINOP struct LinearOperator_s_t

#include "gram_impl.inc"
