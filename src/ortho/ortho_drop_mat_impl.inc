/**
 * @file ortho_drop_mat_impl.inc
 * @brief Type-generic matrix-based B-orthogonalization of U against V
 *
 * For ilobpcg: takes explicit dense matrix instead of LinearOperator.
 * Uses double projection (I - V*V^H*mat)^2*U to handle indefinite metric.
 *
 * Macros required: PREFIX, CTYPE, RTYPE
 */

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

#ifdef CTYPE_IS_COMPLEX
  #define CABS(x) cabs(x)
#else
  #define CABS(x) fabs(x)
#endif

/* Compute ||G-I_sig||_F from upper triangle of Hermitian G (n x n, lda = n)
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
 * ortho_drop_mat - Matrix-based B-orthogonalization of U against V
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
 * @return Number of columns retained in U
 *
 * Algorithm (reference: ilobpcg.c:128-183):
 *   Outer loop (max_outer iters):
 *     1. Double projection: U = U - V*V^H*mat*V*V^H*mat*U  (5 GEMMs)
 *     2. Inner loop (max_inner iters): svqb_mat + check ||U^H*mat*U - I||_F
 *     3. Check ||V^H*mat*U||_F / (||mat*V|| * ||U||) < eps_ortho
 */
uint64_t FN(ortho_drop_mat)(const uint64_t m, const uint64_t n_u,
                                  const uint64_t n_v,
                                  const RTYPE eps_ortho,
                                  const RTYPE eps_drop,
                                  CTYPE *restrict U, CTYPE *restrict V,
                                  CTYPE *restrict mat,
                                  CTYPE *restrict wrk1, CTYPE *restrict wrk2,
                                  CTYPE *restrict wrk3) {
    if (0 == n_u || 0 == n_v) return n_u;

    if (m < n_u + n_v) {
        fprintf(stderr, "ortho_drop_mat: overdetermined m=%lu < n_u+n_v=%lu+%lu\n",
                (unsigned long)m, (unsigned long)n_u, (unsigned long)n_v);
        return n_u;
    }

    /* Compute ||mat*V||_F once */
    FN(gemm_nn)(m, n_v, m, (CTYPE)1, mat, V, (CTYPE)0, wrk1);
    RTYPE MVnorm = FN(nrm2)(m * n_v, wrk1);
    if (MVnorm < eps_ortho) MVnorm = (RTYPE)1;

    const uint64_t max_outer = 3;
    const uint64_t max_inner = 3;

    /* Outer iterations */
    for (uint64_t outer = 0; outer < max_outer; outer++) {

        /* Double projection: U = U - V*V^H*mat*V*V^H*mat*U
         * Computed right-to-left as 5 GEMMs (last fused with subtraction):
         *   Step 1: wrk1 = mat * U              (m x n_u)
         *   Step 2: wrk3 = V^H * wrk1           (n_v x n_u)
         *   Step 3: wrk1 = V * wrk3             (m x n_u)
         *   Step 4: wrk2 = mat * wrk1            (m x n_u)
         *   Step 5: wrk3 = V^H * wrk2           (n_v x n_u)
         *   Step 6: U = U - V * wrk3             (fused gemm_nn with beta=1)
         */

        /* Step 1: wrk1 = mat * U */
        FN(gemm_nn)(m, n_u, m, (CTYPE)1, mat, U, (CTYPE)0, wrk1);

        /* Step 2: wrk3 = V^H * wrk1  (n_v x n_u, fits in wrk3) */
#ifdef CTYPE_IS_COMPLEX
        FN(gemm_hn)(n_v, n_u, m, (CTYPE)1, V, wrk1, (CTYPE)0, wrk3);
#else
        FN(gemm_tn)(n_v, n_u, m, (CTYPE)1, V, wrk1, (CTYPE)0, wrk3);
#endif

        /* Step 3: wrk1 = V * wrk3  (m x n_u) */
        FN(gemm_nn)(m, n_u, n_v, (CTYPE)1, V, wrk3, (CTYPE)0, wrk1);

        /* Step 4: wrk2 = mat * wrk1  (m x n_u) */
        FN(gemm_nn)(m, n_u, m, (CTYPE)1, mat, wrk1, (CTYPE)0, wrk2);

        /* Step 5: wrk3 = V^H * wrk2  (n_v x n_u) */
#ifdef CTYPE_IS_COMPLEX
        FN(gemm_hn)(n_v, n_u, m, (CTYPE)1, V, wrk2, (CTYPE)0, wrk3);
#else
        FN(gemm_tn)(n_v, n_u, m, (CTYPE)1, V, wrk2, (CTYPE)0, wrk3);
#endif

        /* Step 6: U = U - V * wrk3  (fused into single GEMM with beta=1) */
        FN(gemm_nn)(m, n_u, n_v, (CTYPE)(-1), V, wrk3, (CTYPE)1, U);

        /* Inner iterations: svqb_mat + orthonormality check */
        for (uint64_t inner = 0; inner < max_inner; inner++) {
            FN(svqb_mat)(m, n_u, eps_drop, 'n', U, mat, wrk1, wrk2, wrk3);

            /* Check ||U^H*mat*U - I_sig||_F
             * NOTE: denominator choice differs between ortho_drop (||B*U||*||U||)
             * and ortho_indefinite (||U||^2). For indefinite mat, ||mat*U|| can
             * vanish near null-space of mat, making ||mat*U||*||U|| unsuitable.
             * Keeping ||mat*U||*||U|| here for consistency with original code. */
            RTYPE Unorm = FN(nrm2)(m * n_u, U);
            if (Unorm < eps_ortho) Unorm = (RTYPE)1;
            FN(gemm_nn)(m, n_u, m, (CTYPE)1, mat, U, (CTYPE)0, wrk1);
            RTYPE MUnorm = FN(nrm2)(m * n_u, wrk1);
            if (MUnorm < eps_ortho) MUnorm = (RTYPE)1;

#ifdef CTYPE_IS_COMPLEX
            FN(gemm_hn)(n_u, n_u, m, (CTYPE)1, U, wrk1, (CTYPE)0, wrk2);
#else
            FN(gemm_tn)(n_u, n_u, m, (CTYPE)1, U, wrk1, (CTYPE)0, wrk2);
#endif
            RTYPE err = FN(ortho_err_upper)(n_u, wrk2);
            if (err / (Unorm * MUnorm) < eps_ortho) break;
        }

        /* Check ||V^H*mat*U||_F for convergence */
        FN(gemm_nn)(m, n_u, m, (CTYPE)1, mat, U, (CTYPE)0, wrk1);
#ifdef CTYPE_IS_COMPLEX
        FN(gemm_hn)(n_v, n_u, m, (CTYPE)1, V, wrk1, (CTYPE)0, wrk2);
#else
        FN(gemm_tn)(n_v, n_u, m, (CTYPE)1, V, wrk1, (CTYPE)0, wrk2);
#endif
        RTYPE Unorm = FN(nrm2)(m * n_u, U);
        if (Unorm < eps_ortho) Unorm = (RTYPE)1;
        RTYPE VU_err = FN(nrm2)(n_v * n_u, wrk2);
        if (VU_err / (Unorm * MVnorm) < eps_ortho) break;
    }

    return n_u;
}

#undef CONCAT2
#undef CONCAT
#undef FN
#undef MAX
#undef CABS
