#include "lobpcg.h"
#include "lobpcg/linop.h"
#include "lobpcg/blas_wrapper.h"

#define PREFIX z
#define CTYPE c64
#define RTYPE f64
#define LINOP LinearOperator_z_t
#define CTYPE_IS_COMPLEX

#include "indefinite_rr_impl.inc"
