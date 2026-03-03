#include "lobpcg.h"
#include "lobpcg/linop.h"
#include "lobpcg/blas_wrapper.h"

#define PREFIX c
#define CTYPE c32
#define RTYPE f32
#define LINOP LinearOperator_c_t
#define CTYPE_IS_COMPLEX
#define TYPE_IS_FLOAT
#define EPS_TOL 1e-5

#include "lobpcg_impl.inc"
