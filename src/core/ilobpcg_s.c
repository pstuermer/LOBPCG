#include "lobpcg.h"
#include "linop.h"
#include "lobpcg/blas_wrapper.h"

#define PREFIX s
#define CTYPE f32
#define RTYPE f32
#define LINOP LinearOperator_s_t

#include "ilobpcg_impl.inc"
