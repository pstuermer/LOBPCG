#ifndef LOBPCG_H
#define LOPBCG_H

#include "types.h"

#define DECLARE_GET_RESIDUAL(prefix, type, linop) \
  void prefix##_get_residual(const uint64_t, const uin64_t, \
		      type *, type *, type *, type*, linop *, linop*);

TYPE_LIST(DECLARE_GET_RESIDUAL)
#undef DECLARE_GET_RESIDUAL

#define get_residual(size, sizeSub, X, R, eigVal, wrk, A, B) \
  _Generic((X), \
    f32 *: s_get_residual, \
    f64 *: d_get_residual, \
    c32 *: c_get_residual, \
    c64 *: z_get_residual, \
	   )(size, sizeSub, X, R, eigVal, wrk, A, B)
#endif

/*  
// forward declare
struct linop_ctx_t;
struct lobpcg_t

// Generic LOBPCG structure
#define DEFINE_LOBPCG(ctype, rtype, suffix)			\
  typedef struct lobpcg_##suffix##_t lobpcg_##suffix##_t;	\
								\
  struct lobpcg_##suffix##_t {					\
    ctype *restrict S;						\
    ctype *restrict Cx;						\
    ctype *restrict Cp;						\
								\
    ctype *restrict eigVals;					\
    rtype *restrict resNorm;					\
    								\
    ctype *restrict wrk1;					\
    ctype *restrict wrk2;					\
    ctype *restrict wrk3;					\
    ctype *restrict wrk4;					\
    								\
    uint64_t iter;						\
    uint64_t nev;						\
    uint64_t converged;						\
    uint64_t size;						\
    uint64_t sizeSub;						\
    uint64_t maxIter;						\
    								\
    rtype tol;							\
    								\
    struct LinearOperator_##suffix *A;				\
    struct LinearOperator_##suffix *B;				\
  }								\
  

DEFINE_LOBPCG(f32, f32, s)
DEFINE_LOBPCG(f64, f64, d)
DEFINE_LOBPCG(c32, f32, c)
DEFINE_LOBPCG(c64, f64, z)

#define CREATE_LOBPCG(ctype, rtype, suffix)				\
  struct lobpcg_##suffix##_t *lobpcg_##suffix##_alloc() {		\
    lobpcg_##suffix##_t *alg = xmalloc(sizeof(lobpcg_##suffix##_t));	\
									\
    memset(alg, 0, sizeof(lobpcg_##suffix##_t));			\
									\
    return alg;								\
  }									\
									\
									\
  void lobpcg_##suffix##_setup(lobpcg_#suffix##_t *alg,			\
			       const uint64_t size,			\
			       const uint64_t sizeSub,			\
			       const uint64_t nev,			\
			       const uint64_t maxIter,			\
			       const uint64_t mult) {			\
    									\
    if (!alg) {								\
      fprintf(stderr, "Error: alloc lobpcg_t before setting up.\n");	\
      exit(EXIT_FAILURE);						\
    }									\
    									\
    alg->S = xcalloc(mult*size*sizeSub, sizeof(ctype));			\
    alg->Cx = xcalloc(3*sizeSub*sizeSub, sizeof(ctype));		\
    alg->Cp = xcalloc(3*sizeSub*sizeSub, sizeof(ctype));		\
    									\
    alg->eigVals = xcalloc(sizeSub*sizeSub, sizeof(ctype));		\
    alg->resNorm = xcalloc(sizeSub, sizeof(rtype));			\
    									\
    alg->wrk1 = xcalloc(3*mult*size*sizeSub, sizeof(ctype));		\
    alg->wrk2 = xcalloc(3*mult*size*sizeSub, sizeof(ctype));		\
    alg->wrk3 = xcalloc(3*mult*size*sizeSub, sizeof(ctype));		\
    alg->wrk4 = xcalloc(3*mult*size*sizeSub, sizeof(ctype));		\
    									\
    alg->size = size;							\
    alg->sizeSub = sizeSub;						\
    alg->nev = nev;							\
    alg->maxIter = maxIter;						\ 
    alg->mult = mult;							\*/
/* need something for matmuls*//*					\
  }									\


CREATE_LOBPCG(f32, f32, s)
CREATE_LOBPCG(f64, f64, d)
CREATE_LOBPCG(c32, f32, c)
CREATE_LOBPCG(c64, f64, z)

void lobpcg_free(lobpcg_t *alg) {

  if (!alg) return;

  safe_free( alg->S );
  safe_free( alg->Cx );
  safe_free( alg->Cp );

  safe_free( alg->eigVals );
  safe_free( alg->resNorm );

  safe_free( alg->wrk1 );
  safe_free( alg->wrk2 );
  safe_free( alg->wrk3 );
  safe_free( alg->wrk4 );

  safe_free( alg );
}


//#define GET_RESIDUAL(ctype, suffix)		
void zget_residual(const uint64_t size, const uint64_t sizeSub,
		   double complex *X, double complex *R,
		   double complex *eigVal, double complex *wrk,
		   struct LinearOperator_z_t *A,
		   struct LinearOperator_z_t *B) {

  // A*X
  linop_apply_z(A, X, R);

  // X*eigVal*/
  /* need to change this for ColMajor instead of RowMajor */
/*  zgemm_NN(size, sizeSub, sizeSub, X, eigVal, wrk);

  if (B)
    linop_apply_z(B, wrk, NULL);

  zsub_vec(size*sizeSub, R, wrk);

}


#endif // LOBPCG_H
*/
