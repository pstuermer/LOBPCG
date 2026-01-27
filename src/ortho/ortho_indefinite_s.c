/**
 * @file ortho_indefinite_s.c
 * @brief Single precision (float) ortho_indefinite instantiation
 */

#include "../../include/lobpcg/types.h"
#include "../../lobpcg.h"
#include "../../include/lobpcg/blas_wrapper.h"
#include "../../linop.h"

#define PREFIX s
#define CTYPE f32
#define RTYPE f32
#define LINOP struct LinearOperator_s_t

#include "ortho_indefinite_impl.inc"
