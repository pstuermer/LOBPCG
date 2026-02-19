#include "lobpcg.h"
#include "lobpcg/linop.h"
#include "lobpcg/blas_wrapper.h"

#define PREFIX d
#define CTYPE f64
#define RTYPE f64
#define LINOP LinearOperator_d_t

#include "ortho_drop_impl.inc"
