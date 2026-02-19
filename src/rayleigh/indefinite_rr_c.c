#include "lobpcg.h"
#include "lobpcg/linop.h"
#include "lobpcg/blas_wrapper.h"

#define PREFIX c
#define CTYPE c32
#define RTYPE f32
#define LINOP LinearOperator_c_t
#define CTYPE_IS_COMPLEX

#include "indefinite_rr_impl.inc"
