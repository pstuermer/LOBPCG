#include "lobpcg.h"
#include "lobpcg/linop.h"
#include "lobpcg/blas_wrapper.h"

#define PREFIX s
#define CTYPE f32
#define RTYPE f32
#define LINOP LinearOperator_s_t

#include "indefinite_rr_modified_impl.inc"
