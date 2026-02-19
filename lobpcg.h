#ifndef LOBPCG_H
#define LOBPCG_H

#include "types.h"
#include "lobpcg/memory.h"

/* --------------------------------------------------------------------
 * LOBPCG state structure for each type
 * ------------------------------------------------------------------ */
#define LOBPCG_STRUCT(prefix, ctype, rtype, linop)  \
  typedef struct prefix##_lobpcg_t prefix##_lobpcg_t; \
                                                    \
  struct prefix##_lobpcg_t {                        \
    ctype *restrict S;                              \
    ctype *restrict Cx;                             \
    ctype *restrict Cp;                             \
                                                    \
    ctype *restrict AS;                             \
    ctype *restrict BS;                             \
                                                    \
    rtype *restrict eigVals;                        \
    rtype *restrict resNorm;                        \
    int8_t *restrict signature;                     \
                                                    \
    ctype *restrict wrk1;                           \
    ctype *restrict wrk2;                           \
    ctype *restrict wrk3;                           \
    ctype *restrict wrk4;                           \
                                                    \
    uint64_t iter;                                  \
    uint64_t nev;                                   \
    uint64_t converged;                             \
    uint64_t size;                                  \
    uint64_t sizeSub;                               \
    uint64_t maxIter;                               \
                                                    \
    rtype tol;                                      \
                                                    \
    linop *A;                                       \
    linop *B;                                       \
    linop *T;                                       \
  };

TYPE_LIST(LOBPCG_STRUCT)
#undef LOBPCG_STRUCT

/* --------------------------------------------------------------------
 * Function declarations: lobpcg main solver
 * ------------------------------------------------------------------ */
#define DECLARE_LOBPCG(prefix, ctype, rtype, linop) \
  void prefix##_lobpcg(prefix##_lobpcg_t *);

TYPE_LIST(DECLARE_LOBPCG)
#undef DECLARE_LOBPCG

#define lobpcg(alg)             \
  _Generic((alg),               \
    s_lobpcg_t *: s_lobpcg,     \
    d_lobpcg_t *: d_lobpcg,     \
    c_lobpcg_t *: c_lobpcg,     \
    z_lobpcg_t *: z_lobpcg      \
  )(alg)

/* --------------------------------------------------------------------
 * Function declarations: ilobpcg main solver (indefinite LOBPCG)
 * ------------------------------------------------------------------ */
#define DECLARE_ILOBPCG(prefix, ctype, rtype, linop) \
  void prefix##_ilobpcg(prefix##_lobpcg_t *);

TYPE_LIST(DECLARE_ILOBPCG)
#undef DECLARE_ILOBPCG

#define ilobpcg(alg)            \
  _Generic((alg),               \
    s_lobpcg_t *: s_ilobpcg,    \
    d_lobpcg_t *: d_ilobpcg,    \
    c_lobpcg_t *: c_ilobpcg,    \
    z_lobpcg_t *: z_ilobpcg     \
  )(alg)

/* --------------------------------------------------------------------
 * Function declarations: get_residual
 * R = A*X - B*X*eigVal
 * ------------------------------------------------------------------ */
#define DECLARE_GET_RESIDUAL(prefix, ctype, rtype, linop) \
  void prefix##_get_residual(const uint64_t size,         \
                             const uint64_t sizeSub,      \
                             ctype *restrict X,           \
                             ctype *restrict AX,          \
                             ctype *restrict R,           \
                             rtype *restrict eigVal,      \
                             ctype *restrict wrk,         \
                             linop *A,                    \
                             linop *B);

TYPE_LIST(DECLARE_GET_RESIDUAL)
#undef DECLARE_GET_RESIDUAL

#define get_residual(size, sizeSub, X, AX, R, eigVal, wrk, A, B) \
  _Generic((X),                     \
    f32 *: s_get_residual,          \
    f64 *: d_get_residual,          \
    c32 *: c_get_residual,          \
    c64 *: z_get_residual           \
  )(size, sizeSub, X, AX, R, eigVal, wrk, A, B)

/* --------------------------------------------------------------------
 * Function declarations: get_residual_norm
 * Compute relative residual norms
 * ------------------------------------------------------------------ */
#define DECLARE_RESIDUAL_NORM(prefix, ctype, rtype, linop) \
  void prefix##_get_residual_norm(const uint64_t size,     \
                                  const uint64_t nev,      \
                                  ctype *restrict W,       \
                                  rtype *restrict eigVals, \
                                  rtype *restrict resNorm, \
                                  ctype *restrict wrk1,    \
                                  ctype *restrict wrk2,    \
                                  ctype *restrict wrk3,    \
                                  const rtype ANorm,       \
                                  const rtype BNorm,       \
                                  linop *B);

TYPE_LIST(DECLARE_RESIDUAL_NORM)
#undef DECLARE_RESIDUAL_NORM

#define get_residual_norm(size, nev, W, eigVals, resNorm, \
                          wrk1, wrk2, wrk3, ANorm, BNorm, B)       \
  _Generic((W),                         \
    f32 *: s_get_residual_norm,         \
    f64 *: d_get_residual_norm,         \
    c32 *: c_get_residual_norm,         \
    c64 *: z_get_residual_norm          \
  )(size, nev, W, eigVals, resNorm, wrk1, wrk2, wrk3, ANorm, BNorm, B)

/* -------------------------------------------------------------------
 * Function declarations: rayleigh_ritz
 * Standard Rayleigh-Ritz procedure at start
 * ------------------------------------------------------------------ */
#define DECLARE_RAYLEIGH_RITZ(prefix, ctype, rtype, linop) \
  void prefix##_rayleigh_ritz(const uint64_t size,         \
                              const uint64_t sizeSub,      \
                              ctype *restrict S,           \
                              ctype *restrict Cx,          \
                              rtype *restrict eigVal,      \
                              ctype *restrict wrk1,        \
                              ctype *restrict wrk2,        \
                              ctype *restrict wrk3,        \
                              linop *A,                    \
                              linop *B);

TYPE_LIST(DECLARE_RAYLEIGH_RITZ)
#undef DECLARE_RAYLEIGH_RITZ

#define rayleigh_ritz(size, sizeSub, S, Cx, eigVal, wrk1, wrk2, wrk3, A, B) \
  _Generic((S),                     \
    f32 *: s_rayleigh_ritz,         \
    f64 *: d_rayleigh_ritz,         \
    c32 *: c_rayleigh_ritz,         \
    c64 *: z_rayleigh_ritz          \
  )(size, sizeSub, S, Cx, eigVal, wrk1, wrk2, wrk3, A, B)

/* --------------------------------------------------------------------
 * Function declarations: rayleigh_ritz_modified
 * Modified Rayleigh-Ritz procedure for iterations
 * ------------------------------------------------------------------ */
#define DECLARE_RR_MODIFIED(prefix, ctype, rtype, linop)       \
  void prefix##_rayleigh_ritz_modified(const uint64_t size,    \
                                       const uint64_t nx,      \
                                       const uint64_t mult,    \
                                       const uint64_t nconv,   \
                                       const uint64_t ndrop,   \
                                       uint8_t *useOrtho,      \
                                       ctype *restrict S,      \
                                       ctype *restrict wrk1,   \
                                       ctype *restrict wrk2,   \
                                       ctype *restrict wrk3,   \
                                       ctype *restrict Cx,     \
                                       ctype *restrict Cp,     \
                                       rtype *restrict eigVal, \
                                       linop *A,               \
                                       linop *B);

TYPE_LIST(DECLARE_RR_MODIFIED)
#undef DECLARE_RR_MODIFIED

#define rayleigh_ritz_modified(size, nx, mult, nconv, ndrop, useOrtho,       \
                               S, wrk1, wrk2, wrk3, Cx, Cp, eigVal, A, B)   \
  _Generic((S),                             \
    f32 *: s_rayleigh_ritz_modified,        \
    f64 *: d_rayleigh_ritz_modified,        \
    c32 *: c_rayleigh_ritz_modified,        \
    c64 *: z_rayleigh_ritz_modified         \
  )(size, nx, mult, nconv, ndrop, useOrtho, S, wrk1, wrk2, wrk3, Cx, Cp, eigVal, A, B)

/* --------------------------------------------------------------------
 * Function declarations: svqb
 * SVQB orthogonalization (Algorithm from Duersch 2018/Stathopoulos)
 * Returns number of columns after dropping
 * ------------------------------------------------------------------ */
#define DECLARE_SVQB(prefix, ctype, rtype, linop)  \
  uint64_t prefix##_svqb(const uint64_t m,         \
                         const uint64_t n,         \
                         const rtype tau,          \
                         const char drop,          \
                         ctype *restrict U,        \
                         ctype *restrict wrk1,     \
                         ctype *restrict wrk2,     \
                         ctype *restrict wrk3,     \
                         linop *B);

TYPE_LIST(DECLARE_SVQB)
#undef DECLARE_SVQB

#define svqb(m, n, tau, drop, U, wrk1, wrk2, wrk3, B) \
  _Generic((U),                 \
    f32 *: s_svqb,              \
    f64 *: d_svqb,              \
    c32 *: c_svqb,              \
    c64 *: z_svqb               \
  )(m, n, tau, drop, U, wrk1, wrk2, wrk3, B)

/* --------------------------------------------------------------------
 * Function declarations: svqb_mat
 * SVQB orthogonalization with explicit dense matrix (for ilobpcg)
 * Returns number of columns after dropping
 * ------------------------------------------------------------------ */
#define DECLARE_SVQB_MAT(prefix, ctype, rtype, linop) \
  uint64_t prefix##_svqb_mat(const uint64_t m,        \
                             const uint64_t n,         \
                             const rtype tau,          \
                             const char drop,          \
                             ctype *restrict U,        \
                             ctype *restrict mat,      \
                             ctype *restrict wrk1,     \
                             ctype *restrict wrk2,     \
                             ctype *restrict wrk3);

TYPE_LIST(DECLARE_SVQB_MAT)
#undef DECLARE_SVQB_MAT

#define svqb_mat(m, n, tau, drop, U, mat, wrk1, wrk2, wrk3) \
  _Generic((U),                 \
    f32 *: s_svqb_mat,          \
    f64 *: d_svqb_mat,          \
    c32 *: c_svqb_mat,          \
    c64 *: z_svqb_mat           \
  )(m, n, tau, drop, U, mat, wrk1, wrk2, wrk3)

/* --------------------------------------------------------------------
 * Function declarations: ortho_drop
 * Orthogonalize U against V with psd B-inner product
 * Returns number of columns dropped from U
 * ------------------------------------------------------------------ */
#define DECLARE_ORTHO_DROP(prefix, ctype, rtype, linop) \
  uint64_t prefix##_ortho_drop(const uint64_t m,        \
                               const uint64_t n_u,      \
                               const uint64_t n_v,      \
                               const rtype eps_ortho,   \
                               const rtype eps_drop,    \
                               ctype *restrict U,       \
                               ctype *restrict V,       \
                               ctype *restrict wrk1,    \
                               ctype *restrict wrk2,    \
                               ctype *restrict wrk3,    \
                               linop *B);

TYPE_LIST(DECLARE_ORTHO_DROP)
#undef DECLARE_ORTHO_DROP

#define ortho_drop(m, n_u, n_v, eps_ortho, eps_drop, U, V, wrk1, wrk2, wrk3, B) \
  _Generic((U),                     \
    f32 *: s_ortho_drop,            \
    f64 *: d_ortho_drop,            \
    c32 *: c_ortho_drop,            \
    c64 *: z_ortho_drop             \
  )(m, n_u, n_v, eps_ortho, eps_drop, U, V, wrk1, wrk2, wrk3, B)

/* --------------------------------------------------------------------
 * Function declarations: ortho_randomize
 * Orthogonalize U against V with psd B-inner product
 * Same algorithm as ortho_drop but with eps_randomize (looser tolerance)
 * Returns number of columns retained in U
 * ------------------------------------------------------------------ */
#define DECLARE_ORTHO_RANDOMIZE(prefix, ctype, rtype, linop) \
  uint64_t prefix##_ortho_randomize(const uint64_t m,        \
                                     const uint64_t n_u,      \
                                     const uint64_t n_v,      \
                                     const rtype eps_ortho,   \
                                     const rtype eps_randomize, \
                                     ctype *restrict U,       \
                                     ctype *restrict V,       \
                                     ctype *restrict wrk1,    \
                                     ctype *restrict wrk2,    \
                                     ctype *restrict wrk3,    \
                                     linop *B);

TYPE_LIST(DECLARE_ORTHO_RANDOMIZE)
#undef DECLARE_ORTHO_RANDOMIZE

#define ortho_randomize(m, n_u, n_v, eps_ortho, eps_randomize, U, V, wrk1, wrk2, wrk3, B) \
  _Generic((U),                         \
    f32 *: s_ortho_randomize,           \
    f64 *: d_ortho_randomize,           \
    c32 *: c_ortho_randomize,           \
    c64 *: z_ortho_randomize            \
  )(m, n_u, n_v, eps_ortho, eps_randomize, U, V, wrk1, wrk2, wrk3, B)

/* --------------------------------------------------------------------
 * Function declarations: ortho_indefinite
 * B-orthogonalize U against V with indefinite B (for ilobpcg)
 * Returns number of columns dropped from U
 * ------------------------------------------------------------------ */
#define DECLARE_ORTHO_INDEFINITE(prefix, ctype, rtype, linop) \
  uint64_t prefix##_ortho_indefinite(const uint64_t m,        \
                                     const uint64_t n_u,      \
                                     const uint64_t n_v,      \
                                     const rtype eps_ortho,   \
                                     const rtype eps_drop,    \
                                     ctype *restrict U,       \
                                     ctype *restrict V,       \
                                     ctype *restrict sig,     \
                                     ctype *restrict wrk1,    \
                                     ctype *restrict wrk2,    \
                                     ctype *restrict wrk3,    \
                                     linop *B);

TYPE_LIST(DECLARE_ORTHO_INDEFINITE)
#undef DECLARE_ORTHO_INDEFINITE

#define ortho_indefinite(m, n_u, n_v, eps_ortho, eps_drop, U, V, sig, wrk1, wrk2, wrk3, B) \
  _Generic((U),                         \
    f32 *: s_ortho_indefinite,          \
    f64 *: d_ortho_indefinite,          \
    c32 *: c_ortho_indefinite,          \
    c64 *: z_ortho_indefinite           \
  )(m, n_u, n_v, eps_ortho, eps_drop, U, V, sig, wrk1, wrk2, wrk3, B)

/* --------------------------------------------------------------------
 * Function declarations: ortho_randomized_mat
 * Matrix-based B-orthogonalization of U against V (for ilobpcg)
 * Uses double projection (I - V*V^H*mat)^2 for indefinite metric
 * Returns number of columns retained in U
 * ------------------------------------------------------------------ */
#define DECLARE_ORTHO_RMAT(prefix, ctype, rtype, linop)       \
  uint64_t prefix##_ortho_randomized_mat(const uint64_t m,    \
                                         const uint64_t n_u,  \
                                         const uint64_t n_v,  \
                                         const rtype eps_ortho, \
                                         const rtype eps_drop,  \
                                         ctype *restrict U,    \
                                         ctype *restrict V,    \
                                         ctype *restrict mat,  \
                                         ctype *restrict wrk1, \
                                         ctype *restrict wrk2, \
                                         ctype *restrict wrk3);

TYPE_LIST(DECLARE_ORTHO_RMAT)
#undef DECLARE_ORTHO_RMAT

#define ortho_randomized_mat(m, n_u, n_v, eps_ortho, eps_drop, U, V, mat, wrk1, wrk2, wrk3) \
  _Generic((U),                             \
    f32 *: s_ortho_randomized_mat,          \
    f64 *: d_ortho_randomized_mat,          \
    c32 *: c_ortho_randomized_mat,          \
    c64 *: z_ortho_randomized_mat           \
  )(m, n_u, n_v, eps_ortho, eps_drop, U, V, mat, wrk1, wrk2, wrk3)

/* --------------------------------------------------------------------
 * Function declarations: indefinite_rayleigh_ritz
 * Indefinite Rayleigh-Ritz using GGEV (for ilobpcg)
 * ------------------------------------------------------------------ */
#define DECLARE_INDEF_RR(prefix, ctype, rtype, linop)                \
  void prefix##_indefinite_rayleigh_ritz(                            \
      const uint64_t size, const uint64_t sizeSub,                   \
      ctype *restrict S, ctype *restrict Cx, rtype *restrict eigVal, \
      int8_t *restrict signature,                                    \
      ctype *restrict wrk1, ctype *restrict wrk2,                   \
      ctype *restrict wrk3, ctype *restrict wrk4,                   \
      linop *A, linop *B);

TYPE_LIST(DECLARE_INDEF_RR)
#undef DECLARE_INDEF_RR

#define indefinite_rayleigh_ritz(size, sizeSub, S, Cx, eigVal, sig,   \
                                  wrk1, wrk2, wrk3, wrk4, A, B)       \
  _Generic((S),                                 \
    f32 *: s_indefinite_rayleigh_ritz,          \
    f64 *: d_indefinite_rayleigh_ritz,          \
    c32 *: c_indefinite_rayleigh_ritz,          \
    c64 *: z_indefinite_rayleigh_ritz           \
  )(size, sizeSub, S, Cx, eigVal, sig, wrk1, wrk2, wrk3, wrk4, A, B)

/* --------------------------------------------------------------------
 * Function declarations: indefinite_rayleigh_ritz_modified
 * Modified indefinite Rayleigh-Ritz with Cx/Cp extraction (for ilobpcg)
 * ------------------------------------------------------------------ */
#define DECLARE_INDEF_RR_MOD(prefix, ctype, rtype, linop)                \
  void prefix##_indefinite_rayleigh_ritz_modified(                       \
      const uint64_t size, const uint64_t nx,                            \
      const uint64_t mult,                                               \
      const uint64_t nconv, const uint64_t ndrop,                        \
      ctype *restrict S, ctype *restrict wrk1,                           \
      ctype *restrict wrk2, ctype *restrict wrk3, ctype *restrict wrk4, \
      ctype *restrict Cx, ctype *restrict Cp,                            \
      rtype *restrict eigVal, int8_t *restrict signature,                \
      linop *A, linop *B);

TYPE_LIST(DECLARE_INDEF_RR_MOD)
#undef DECLARE_INDEF_RR_MOD

#define indefinite_rayleigh_ritz_modified(size, nx, mult, nconv, ndrop,     \
                                          S, wrk1, wrk2, wrk3, wrk4,       \
                                          Cx, Cp, eigVal, sig, A, B)        \
  _Generic((S),                                         \
    f32 *: s_indefinite_rayleigh_ritz_modified,         \
    f64 *: d_indefinite_rayleigh_ritz_modified,         \
    c32 *: c_indefinite_rayleigh_ritz_modified,         \
    c64 *: z_indefinite_rayleigh_ritz_modified          \
  )(size, nx, mult, nconv, ndrop, S, wrk1, wrk2, wrk3, wrk4, Cx, Cp, eigVal, sig, A, B)

/* --------------------------------------------------------------------
 * Function declarations: Gram matrix helpers
 * ------------------------------------------------------------------ */
#define DECLARE_APPLY_BLOCK_OP(prefix, ctype, rtype, linop) \
  void prefix##_apply_block_op(const linop *Op, ctype *restrict X, \
                               ctype *restrict Y, const uint64_t n, \
                               const uint64_t k);

TYPE_LIST(DECLARE_APPLY_BLOCK_OP)
#undef DECLARE_APPLY_BLOCK_OP

#define apply_block_op(Op, X, Y, n, k) \
  _Generic((X),                        \
    f32 *: s_apply_block_op, \
    f64 *: d_apply_block_op, \
    c32 *: c_apply_block_op, \
    c64 *: z_apply_block_op  \
  )(Op, X, Y, n, k)

#define DECLARE_GRAM_SELF(prefix, ctype, rtype, linop) \
  void prefix##_gram_self(ctype *restrict U, const uint64_t n,       \
                          const uint64_t k, const linop *B,           \
                          ctype *restrict G, const uint64_t ldg,      \
                          ctype *restrict wrk);

TYPE_LIST(DECLARE_GRAM_SELF)
#undef DECLARE_GRAM_SELF

#define gram_self(U, n, k, B, G, ldg, wrk) \
  _Generic((U),                            \
    f32 *: s_gram_self, \
    f64 *: d_gram_self, \
    c32 *: c_gram_self, \
    c64 *: z_gram_self  \
  )(U, n, k, B, G, ldg, wrk)

#define DECLARE_GRAM_CROSS(prefix, ctype, rtype, linop) \
  void prefix##_gram_cross(ctype *restrict V, const uint64_t nv,      \
                           ctype *restrict U, const uint64_t nu,      \
                           const uint64_t n, const linop *B,           \
                           ctype *restrict G, const uint64_t ldg,      \
                           ctype *restrict wrk);

TYPE_LIST(DECLARE_GRAM_CROSS)
#undef DECLARE_GRAM_CROSS

#define gram_cross(V, nv, U, nu, n, B, G, ldg, wrk) \
  _Generic((V),                                      \
    f32 *: s_gram_cross, \
    f64 *: d_gram_cross, \
    c32 *: c_gram_cross, \
    c64 *: z_gram_cross  \
  )(V, nv, U, nu, n, B, G, ldg, wrk)

/* --------------------------------------------------------------------
 * Function declarations: fill_random
 * Fill array with uniform random values
 * ------------------------------------------------------------------ */
#define DECLARE_FILL_RANDOM(prefix, ctype, rtype, linop) \
  void prefix##_fill_random(uint64_t n, ctype *x);

TYPE_LIST(DECLARE_FILL_RANDOM)
#undef DECLARE_FILL_RANDOM

#define fill_random(n, x)               \
  _Generic((x),                         \
    f32 *: s_fill_random,               \
    f64 *: d_fill_random,               \
    c32 *: c_fill_random,               \
    c64 *: z_fill_random                \
  )(n, x)

/* --------------------------------------------------------------------
 * Function declarations: estimate_norm
 * Estimate spectral radius via power iteration
 * ------------------------------------------------------------------ */
#define DECLARE_ESTIMATE_NORM(prefix, ctype, rtype, linop) \
  rtype prefix##_estimate_norm(uint64_t size, linop *A,    \
                               ctype *wrk1, ctype *wrk2);

TYPE_LIST(DECLARE_ESTIMATE_NORM)
#undef DECLARE_ESTIMATE_NORM

#define estimate_norm(size, A, wrk1, wrk2)  \
  _Generic((wrk1),                          \
    f32 *: s_estimate_norm,                 \
    f64 *: d_estimate_norm,                 \
    c32 *: c_estimate_norm,                 \
    c64 *: z_estimate_norm                  \
  )(size, A, wrk1, wrk2)

/* --------------------------------------------------------------------
 * K-based Rayleigh-Ritz: aliases to standard RR
 * (algorithmically identical — both use Cholesky + HEEV)
 * ------------------------------------------------------------------ */
#define s_krayleigh_ritz           s_rayleigh_ritz
#define d_krayleigh_ritz           d_rayleigh_ritz
#define c_krayleigh_ritz           c_rayleigh_ritz
#define z_krayleigh_ritz           z_rayleigh_ritz

#define s_krayleigh_ritz_modified  s_rayleigh_ritz_modified
#define d_krayleigh_ritz_modified  d_rayleigh_ritz_modified
#define c_krayleigh_ritz_modified  c_rayleigh_ritz_modified
#define z_krayleigh_ritz_modified  z_rayleigh_ritz_modified

#define krayleigh_ritz(size, sizeSub, S, Cx, eigVal, wrk1, wrk2, wrk3, A, B) \
  rayleigh_ritz(size, sizeSub, S, Cx, eigVal, wrk1, wrk2, wrk3, A, B)

#define krayleigh_ritz_modified(size, nx, mult, nconv, ndrop, useOrtho,       \
                                S, wrk1, wrk2, wrk3, Cx, Cp, eigVal, A, B)   \
  rayleigh_ritz_modified(size, nx, mult, nconv, ndrop, useOrtho,              \
                         S, wrk1, wrk2, wrk3, Cx, Cp, eigVal, A, B)

/* --------------------------------------------------------------------
 * lobpcg_alloc / lobpcg_free
 * Allocate and free LOBPCG state structures
 * ------------------------------------------------------------------ */
#define DEFINE_LOBPCG_ALLOC(prefix, ctype, rtype, linop)		\
  static inline prefix##_lobpcg_t *prefix##_lobpcg_alloc(		\
			  uint64_t n, uint64_t nev, uint64_t sizeSub) { \
    prefix##_lobpcg_t *alg = xcalloc(1, sizeof(prefix##_lobpcg_t));	\
    alg->size = n;							\
    alg->sizeSub = sizeSub;						\
    alg->nev = nev;							\
    alg->sizeSub = nev;							\
    alg->S       = xcalloc(n * 3 * sizeSub, sizeof(ctype));		\
    alg->Cx      = xcalloc(3 * sizeSub * sizeSub, sizeof(ctype));	\
    alg->Cp      = xcalloc(3 * sizeSub * sizeSub, sizeof(ctype));	\
    alg->eigVals = xcalloc(sizeSub, sizeof(rtype));			\
    alg->resNorm = xcalloc(sizeSub, sizeof(rtype));			\
    alg->wrk1    = xcalloc(n * 3 * sizeSub, sizeof(ctype));		\
    alg->wrk2    = xcalloc(n * 3 * sizeSub, sizeof(ctype));		\
    alg->wrk3    = xcalloc(n * 3 * sizeSub, sizeof(ctype));		\
    alg->wrk4    = xcalloc(n * 3 * sizeSub, sizeof(ctype));		\
    return alg;								\
  }

TYPE_LIST(DEFINE_LOBPCG_ALLOC)
#undef DEFINE_LOBPCG_ALLOC

#define lobpcg_alloc(n, nev, sizeSub, prefix) \
  prefix##_lobpcg_alloc(n, nev, sizeSub)

#define DEFINE_LOBPCG_FREE(prefix, ctype, rtype, linop)                     \
  static inline void prefix##_lobpcg_free(prefix##_lobpcg_t **alg) {        \
    if (!alg || !*alg) return;                                              \
    safe_free((void**)&(*alg)->S);                                          \
    safe_free((void**)&(*alg)->Cx);                                         \
    safe_free((void**)&(*alg)->Cp);                                         \
    safe_free((void**)&(*alg)->AS);                                         \
    safe_free((void**)&(*alg)->BS);                                         \
    safe_free((void**)&(*alg)->eigVals);                                    \
    safe_free((void**)&(*alg)->resNorm);                                    \
    safe_free((void**)&(*alg)->signature);                                  \
    safe_free((void**)&(*alg)->wrk1);                                       \
    safe_free((void**)&(*alg)->wrk2);                                       \
    safe_free((void**)&(*alg)->wrk3);                                       \
    safe_free((void**)&(*alg)->wrk4);                                       \
    safe_free((void**)alg);                                                 \
  }

TYPE_LIST(DEFINE_LOBPCG_FREE)
#undef DEFINE_LOBPCG_FREE

#define lobpcg_free(alg) _Generic((*alg),   \
    s_lobpcg_t *: s_lobpcg_free,            \
    d_lobpcg_t *: d_lobpcg_free,            \
    c_lobpcg_t *: c_lobpcg_free,            \
    z_lobpcg_t *: z_lobpcg_free             \
  )(alg)

/* --------------------------------------------------------------------
 * ilobpcg_alloc
 * Allocate LOBPCG state with extra buffers for indefinite solver
 * ------------------------------------------------------------------ */
#define DEFINE_ILOBPCG_ALLOC(prefix, ctype, rtype, linop)		\
  static inline prefix##_lobpcg_t *prefix##_ilobpcg_alloc(		\
      		      uint64_t n, uint64_t nev, uint64_t sizeSub) {     \
    prefix##_lobpcg_t *alg = prefix##_lobpcg_alloc(n, nev, sizeSub);	\
    /* Reallocate Cp larger for indefinite: 3*nev × 2*nev */		\
    safe_free((void**)&alg->Cp);					\
    alg->Cp        = xcalloc(3*3*sizeSub*sizeSub, sizeof(ctype));	\
    alg->AS        = xcalloc(3*n*sizeSub, sizeof(ctype));		\
    alg->signature = xcalloc(3*sizeSub, sizeof(int8_t));		\
    return alg;								\
  }

TYPE_LIST(DEFINE_ILOBPCG_ALLOC)
#undef DEFINE_ILOBPCG_ALLOC

#define ilobpcg_alloc(n, nev, prefix) prefix##_ilobpcg_alloc(n, nev)

#endif /* LOBPCG_H */
