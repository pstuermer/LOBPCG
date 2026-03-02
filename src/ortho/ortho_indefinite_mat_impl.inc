/**
 * @file ortho_indefinite_mat_impl.inc
 * @brief Type-generic matrix-based B-orthogonalization of U against V
 *
 * For ilobpcg: takes explicit dense matrix instead of LinearOperator.
 * Uses double projection (I - V*V^H*mat)^2*U to handle indefinite metric.
 *
 * Macros required: PREFIX, CTYPE, RTYPE
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "lobpcg/memory.h"

#define CONCAT2(a, b) a##_##b
#define CONCAT(a, b) CONCAT2(a, b)
#define FN(name) CONCAT(PREFIX, name)

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifdef TYPE_IS_FLOAT
  #ifdef CTYPE_IS_COMPLEX
  #define CABS(x) cabsf(x)
  #else
  #define CABS(x) fabsf(x)
  #endif
#else
  #ifdef CTYPE_IS_COMPLEX
  #define CABS(x) cabs(x)
  #else
  #define CABS(x) fabs(x)
  #endif
#endif

/* Compute ||G - I_sig||_F from upper triangle of Hermitian G (n x n, lda = n)
 * After SVQB with indefinite mat, diag(G) ~ +/-1, so check |G_jj| ~ 1 */
static RTYPE FN(ortho_err_upper)(uint64_t n, const CTYPE *G) {
  RTYPE sum = 0;
  for (uint64_t j = 0; j < n; j++) {
    const RTYPE d = CABS(G[j + j*n]) - (RTYPE)1;
    sum += d*d;
    for (uint64_t i = 0; i < j; i++) {
      const RTYPE v = CABS(G[i + j*n]);
      sum += 2*v*v;
    }
  }
  return sqrt(sum);
}

/**
 * ortho_indefinite_mat - Matrix-based B-orthogonalization of U against V
 *
 * @param m          Number of rows (problem size)
 * @param n_u        Number of columns of U
 * @param n_v        Number of columns of V (V assumed mat-orthonormal)
 * @param eps_ortho  Tolerance for orthogonality checks
 * @param eps_drop   Drop tolerance passed to svqb_mat
 * @param U          [in/out] m x n_u, mat-orthonormalized against V on output
 * @param V          [in] m x n_v, mat-orthonormal basis
 * @param mat        [in] m x m dense matrix (indefinite metric)
 * @param wrk1       Workspace >= m * max(n_u, n_v)
 * @param wrk2       Workspace >= m * max(n_u, n_v)
 * @param wrk3       Workspace >= m * max(n_u, n_v)
 *
 * @return Number of columns dropped from U (always 0 for now)
 *
 * Algorithm (reference: ilobpcg.c:128-183):
 *   Outer loop (max_outer iters):
 *     1. Double projection: U = U - V*V^H*mat*V*V^H*mat*U  (via gram_cross_mat)
 *     2. Inner loop (max_inner iters): svqb_mat + check ||U^H*mat*U - I||_F
 *     3. Check ||V^H*mat*U||_F / (||mat*V|| * ||U||) < eps_ortho
 */
uint64_t FN(ortho_indefinite_mat)(const uint64_t m, const uint64_t n_u,
                                  const uint64_t n_v,
                                  const RTYPE eps_ortho,
                                  const RTYPE eps_drop,
                                  CTYPE *restrict U, CTYPE *restrict V,
                                  CTYPE *restrict mat,
                                  CTYPE *restrict wrk1, CTYPE *restrict wrk2,
                                  CTYPE *restrict wrk3) {
  if (0 == n_u || 0 == n_v) return 0;

  if (m < n_u + n_v) {
    fprintf(stderr, "ortho_indefinite_mat: overdetermined m=%lu < n_u+n_v=%lu+%lu\n",
            (unsigned long)m, (unsigned long)n_u, (unsigned long)n_v);
    return 0;
  }

  /* Compute ||mat*V||_F once */
  FN(gemm_nn)(m, n_v, m, (CTYPE)1, mat, V, (CTYPE)0, wrk1);
  RTYPE MV_norm = FN(nrm2)(m * n_v, wrk1);
  if (MV_norm < eps_ortho) MV_norm = (RTYPE)1;

  const uint64_t max_outer = 3;
  const uint64_t max_inner = 3;

  /* Outer iterations: project out V component and re-orthonormalize */
  for (uint64_t outer = 0; outer < max_outer; outer++) {

    /* Double projection: U = U - V*V^H*mat*V*V^H*mat*U
     * Computed right-to-left as:
     *   Steps 1-2: wrk3 = V^H * mat * U    (n_v x n_u)
     *   Step 3:    wrk1 = V * wrk3          (m x n_u)
     *   Steps 4-5: wrk3 = V^H * mat * wrk1  (n_v x n_u)
     *   Step 6:    U = U - V * wrk3          (fused gemm with beta=1)
     */

    /* Steps 1-2: wrk3 = V^H * mat * U  (n_v x n_u) */
    FN(gram_cross_mat)(V, n_v, U, n_u, m, mat, wrk3, n_v, wrk1);

    /* Step 3: wrk1 = V * wrk3  (m x n_u) */
    FN(gemm_nn)(m, n_u, n_v, (CTYPE)1, V, wrk3, (CTYPE)0, wrk1);

    /* Steps 4-5: wrk3 = V^H * mat * wrk1  (n_v x n_u) */
    FN(gram_cross_mat)(V, n_v, wrk1, n_u, m, mat, wrk3, n_v, wrk2);

    /* Step 6: U = U - V * wrk3  (fused into single GEMM with beta=1) */
    FN(gemm_nn)(m, n_u, n_v, (CTYPE)(-1), V, wrk3, (CTYPE)1, U);

    /* Inner iterations: svqb_mat + orthonormality check */
    for (uint64_t inner = 0; inner < max_inner; inner++) {
      FN(svqb_mat)(m, n_u, eps_drop, 'n', U, mat, wrk1, wrk2, wrk3);

      /* Check ||U^H*mat*U - I_sig||_F / ||U||^2 */
      FN(gram_self_mat)(U, m, n_u, mat, wrk1, n_u, wrk2);
      RTYPE ortho_err = FN(ortho_err_upper)(n_u, wrk1);
      RTYPE U_norm = FN(nrm2)(m * n_u, U);
      if (U_norm < eps_ortho) U_norm = (RTYPE)1;

      if (ortho_err / (U_norm * U_norm) < eps_ortho) break;
    }

    /* Check ||V^H*mat*U||_F for convergence */
    FN(gram_cross_mat)(V, n_v, U, n_u, m, mat, wrk3, n_v, wrk1);

    RTYPE U_norm = FN(nrm2)(m * n_u, U);
    if (U_norm < eps_ortho) U_norm = (RTYPE)1;

    RTYPE VU_err = FN(nrm2)(n_v * n_u, wrk3) / (MV_norm * U_norm);
    if (VU_err < eps_ortho) break;
  }

  return 0;  /* No column dropping implemented yet */
}

#undef CONCAT2
#undef CONCAT
#undef FN
#undef MAX
#undef CABS
