#include "lobpcg.h"
#include "lobpcg/linop.h"
#include "lobpcg/blas_wrapper.h"

#define PREFIX s
#define CTYPE f32
#define RTYPE f32
#define LINOP LinearOperator_s_t
#define TYPE_IS_FLOAT

#include "ortho_drop_impl.inc"
