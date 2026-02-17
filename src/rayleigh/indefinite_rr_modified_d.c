#include "lobpcg.h"
#include "linop.h"
#include "lobpcg/blas_wrapper.h"

#define PREFIX d
#define CTYPE f64
#define RTYPE f64
#define LINOP LinearOperator_d_t

#include "indefinite_rr_modified_impl.inc"
