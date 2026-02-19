/**
 * @file svqb_s.c
 * @brief Single precision (float) SVQB instantiation
 */

#include "../../include/lobpcg/types.h"
#include "../../lobpcg.h"
#include "../../include/lobpcg/blas_wrapper.h"
#include "lobpcg/linop.h"

#define PREFIX s
#define CTYPE f32
#define RTYPE f32
#define LINOP struct LinearOperator_s_t

#include "svqb_impl.inc"
