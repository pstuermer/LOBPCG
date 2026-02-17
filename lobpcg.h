#ifndef LOBPCG_H
#define LOBPCG_H

#include "types.h"

// Forward declarations
struct linop_ctx_t;
struct lobpcg_t;

// Forward declaration for linear operator types
struct LinearOperator_s;
struct LinearOperator_d;
struct LinearOperator_c;
struct LinearOperator_z;

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
    ctype *restrict eigVals;                        \
    rtype *restrict resNorm;                        \
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
 * Function declarations: get_residual
 * R = A*X - B*X*eigVal
 * ------------------------------------------------------------------ */
#define DECLARE_GET_RESIDUAL(prefix, ctype, rtype, linop) \
  void prefix##_get_residual(const uint64_t size,         \
                             const uint64_t sizeSub,      \
                             ctype *restrict X,           \
                             ctype *restrict AX,          \
                             ctype *restrict R,           \
                             ctype *restrict eigVal,      \
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
                                  const uint64_t sizeSub,  \
                                  const uint64_t nev,      \
                                  ctype *restrict W,       \
                                  ctype *restrict eigVals, \
                                  rtype *restrict resNorm, \
                                  ctype *restrict wrk1,    \
                                  ctype *restrict wrk2,    \
                                  ctype *restrict wrk3,    \
                                  const rtype ANorm,       \
                                  const rtype BNorm,       \
                                  linop *B);

TYPE_LIST(DECLARE_RESIDUAL_NORM)
#undef DECLARE_RESIDUAL_NORM

#define get_residual_norm(size, sizeSub, nev, W, eigVals, resNorm, \
                          wrk1, wrk2, wrk3, ANorm, BNorm, B)       \
  _Generic((W),                         \
    f32 *: s_get_residual_norm,         \
    f64 *: d_get_residual_norm,         \
    c32 *: c_get_residual_norm,         \
    c64 *: z_get_residual_norm          \
  )(size, sizeSub, nev, W, eigVals, resNorm, wrk1, wrk2, wrk3, ANorm, BNorm, B)

/* -------------------------------------------------------------------
 * Function declarations: rayleigh_ritz
 * Standard Rayleigh-Ritz procedure at start
 * ------------------------------------------------------------------ */
#define DECLARE_RAYLEIGH_RITZ(prefix, ctype, rtype, linop) \
  void prefix##_rayleigh_ritz(const uint64_t size,         \
                              const uint64_t sizeSub,      \
                              ctype *restrict S,           \
                              ctype *restrict Cx,          \
                              ctype *restrict eigVal,      \
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

#endif /* LOBPCG_H */
