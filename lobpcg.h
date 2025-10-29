#ifndef LOBPCG_H
#define LOPBCG_H

#include "types.h"

// still missing the struct and allocation and stuff

#define DECLARE_LOBPCG(prefix, ctype, rtype, linop) \
  void prefix##_lobpcg(lobpcg_##prefix##_t *);

TYPE_LIST(DECLARE_LOBPCG)
#undef DECLARE_LOBPCG

#define lobpcg(alg) \
  _Generic((alg),   \
    lobpcg_s_t *: s_lobpcg, \
    lobpcg_d_t *: d_lobpcg, \
    lobpcg_c_t *: c_lobpcg, \
	   lobpcg_z_t *: z_lobpcg		\
	   )(alg)

/* ------------------------------------------------------- */

#define DECLARE_GET_RESIDUAL(prefix, ctype, rtype, linop)		\
  void prefix##_get_residual(const uint64_t, const uint64_t,		\
			     ctype *restrict, ctype *restrict,		\
			     ctype *restrict, ctype *restrict,		\
			     linop *, linop*);

TYPE_LIST(DECLARE_GET_RESIDUAL)
#undef DECLARE_GET_RESIDUAL

#define get_residual(size, sizeSub, X, R, eigVal, wrk, A, B)	\
  _Generic((X),							\
	   f32 *: s_get_residual,				\
	   f64 *: d_get_residual,				\
	   c32 *: c_get_residual,				\
	   c64 *: z_get_residual				\
	   )(size, sizeSub, X, R, eigVal, wrk, A, B)
#endif

/* ------------------------------------------------------- */

#define DECLARE_RESIDUAL_NORM(prefix, ctype, rtype, linop)		\
  void prefix##_get_residual_norm(const uint64_t, const uint64_t, const uint64_t, \
				  ctype *restrict, ctype *restrict, rtype *restrict, \
				  ctype *restrict, ctype *restrict,	\
				  const rtype, const rtype, linop*);

TYPE_LIST(DECLARE_RESIDUAL_NORM)
#undef

#define get_residual_norm(size, sizeSub, nev, W, eigVals, resNorm, \
			  wrk1, wrk2, wrk3, ANorm, Bnorm, B) \
  _Generic((W),						     \
    f32 *: s_get_residual_norm, \
    f64 *: d_get_residual_norm, \
    c32 *: c_get_residual_norm, \
    c64 *: z_get_residual_norm \
	   )(size, sizeSub, nev, W, eigVals, resNorm, wrk1, wrk2, \
	     wrk3, Anorm, Bnorm, B)

/* ------------------------------------------------------- */

#define DECLARE_RAYLEIGH_RITZ(prefix, ctype, rtype, linop)	\
  void prefix##_rayleigh_ritz(const uint64_t, const uint64_t,	\
			      ctype *restrict, ctype *restrict, \
			      ctype *restrict, ctype *restrict, \
			      ctype *restrict, ctype *restrict, \
			      linop *, linop *);

TYPE_LIST(DECLARE_RAYLEIGH_RITZ)
#undef


#define rayleigh_ritz(size, sizeSub, S, Cx, eigVal,	     \
		      wrk1, wrk2, wrk3, A, B)		     \
  _Generic((S),						     \
	   f32 *: s_rayleigh_ritz,			     \
	   f64 *: d_rayleigh_ritz,			     \
	   c32 *: c_rayleigh_ritz,			     \
	   c64 *: z_rayleigh_ritz			     \
	   )(size, sizeSub, S, Cx, eigVal, wrk1, wrk2,	     \
	     wrk3, A, B)

/* ------------------------------------------------------------------------ */

void zrayleigh_ritz_modified(const uint64_t, const uint64_t,
			     const uint64_t, uint8_t *,
			     f64, c64 *restrict, c64 *restrict,
			     c64 *restrict, c64 *restrict,
			     c64 *restrict, f64 *restrict,
			     linop *, linop *);

void zsvqb(const uint64_t, const uint64_t, const f64, const char,
	   c64 *restrict, c64 *restrict, c64 *restrict, c64 *restrict,
	   linop *);

void zortho_drop(const uint64_t, const uint64_t, const uint64_t,
		 const uint64_t, const f64, c64 *restrict,
		 c64 *restrict, c64 *restrict, c64 *restrict,
		 c64 *restrict, linop *);

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
