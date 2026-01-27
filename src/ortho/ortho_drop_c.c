#include "lobpcg.h"
#include "linop.h"
#include "lobpcg/blas_wrapper.h"

#define PREFIX c
#define CTYPE c32
#define RTYPE f32
#define LINOP LinearOperator_c_t
#define CTYPE_IS_COMPLEX

#include "ortho_drop_impl.inc"
