/**
 * @file ortho_randomized_mat_impl.inc
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
  #define CREAL(x) creal(x)
#else
  #define CABS(x) fabs(x)
  #define CREAL(x) (x)
#endif

/**
 * ortho_randomized_mat - Matrix-based B-orthogonalization of U against V
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
 * @param wrk3       Workspace >= max(n_v * n_u, n_u * n_u)
 *
 * @return Number of columns retained in U
 *
 * Algorithm (reference: ilobpcg.c:128-183):
 *   Outer loop (3 iters):
 *     1. Double projection: U = (I - V*V^H*mat*V*V^H*mat)*U
 *     2. Inner loop (3 iters): svqb_mat + check ||U^H*mat*U - I||_F
 *     3. Check ||V^H*mat*U||_F / (||mat*V|| * ||U||) < eps_ortho
 */
uint64_t FN(ortho_randomized_mat)(const uint64_t m, const uint64_t n_u,
                                  const uint64_t n_v,
                                  const RTYPE eps_ortho,
                                  const RTYPE eps_drop,
                                  CTYPE *restrict U, CTYPE *restrict V,
                                  CTYPE *restrict mat,
                                  CTYPE *restrict wrk1, CTYPE *restrict wrk2,
                                  CTYPE *restrict wrk3) {
    if (0 == n_u || 0 == n_v) return n_u;

    RTYPE Unorm = 0, MUnorm = 0, MVnorm = 0;

    /* Compute ||mat*V||_F once */
    FN(gemm_nn)(m, n_v, m, (CTYPE)1, mat, V, (CTYPE)0, wrk1);
    MVnorm = FN(nrm2)(m * n_v, wrk1);
    if (MVnorm < eps_ortho) MVnorm = (RTYPE)1;

    /* Outer iterations */
    for (uint64_t outer = 0; outer < 3; outer++) {

        /* Double projection: U = (I - V*V^H*mat*V*V^H*mat) * U
         * Computed right-to-left as 6 GEMMs:
         *   wrk1 = mat * U                (m x n_u)
         *   wrk3 = V^H * wrk1            (n_v x n_u)
         *   wrk1 = V * wrk3              (m x n_u)
         *   wrk3_big = mat * wrk1         (m x n_u)  -- reuse wrk2 for big result
         *   wrk3 = V^H * wrk3_big        (n_v x n_u)
         *   wrk1 = V * wrk3              (m x n_u)   -- this is the projection
         *   U = U - wrk1
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

        /* Step 4: wrk2 = mat * wrk1  (m x n_u, use wrk2 as temp) */
        FN(gemm_nn)(m, n_u, m, (CTYPE)1, mat, wrk1, (CTYPE)0, wrk2);

        /* Step 5: wrk3 = V^H * wrk2  (n_v x n_u) */
#ifdef CTYPE_IS_COMPLEX
        FN(gemm_hn)(n_v, n_u, m, (CTYPE)1, V, wrk2, (CTYPE)0, wrk3);
#else
        FN(gemm_tn)(n_v, n_u, m, (CTYPE)1, V, wrk2, (CTYPE)0, wrk3);
#endif

        /* Step 6: wrk1 = V * wrk3  (m x n_u) -- the projection */
        FN(gemm_nn)(m, n_u, n_v, (CTYPE)1, V, wrk3, (CTYPE)0, wrk1);

        /* U = U - projection */
        for (uint64_t i = 0; i < m * n_u; i++)
            U[i] -= wrk1[i];

        /* Inner iterations: svqb_mat + orthonormality check */
        for (uint64_t inner = 0; inner < 3; inner++) {
            FN(svqb_mat)(m, n_u, eps_drop, 'n', U, mat, wrk1, wrk2, wrk3);

            /* Check ||U^H*mat*U - I||_F */
            Unorm = FN(nrm2)(m * n_u, U);
            FN(gemm_nn)(m, n_u, m, (CTYPE)1, mat, U, (CTYPE)0, wrk1);
            MUnorm = FN(nrm2)(m * n_u, wrk1);

#ifdef CTYPE_IS_COMPLEX
            FN(gemm_hn)(n_u, n_u, m, (CTYPE)1, U, wrk1, (CTYPE)0, wrk2);
#else
            FN(gemm_tn)(n_u, n_u, m, (CTYPE)1, U, wrk1, (CTYPE)0, wrk2);
#endif
            /* wrk2[i,i] = |Re(wrk2[i,i])| - 1 to handle indefinite signs */
            for (uint64_t k = 0; k < n_u; k++)
                wrk2[k + k*n_u] = (CTYPE)(fabs(CREAL(wrk2[k + k*n_u])) - 1.0);

            RTYPE err = FN(nrm2)(n_u * n_u, wrk2);
            if (err / (Unorm * MUnorm) < eps_ortho) break;
        }

        /* Check ||V^H*mat*U||_F for convergence */
        FN(gemm_nn)(m, n_u, m, (CTYPE)1, mat, U, (CTYPE)0, wrk1);
#ifdef CTYPE_IS_COMPLEX
        FN(gemm_hn)(n_v, n_u, m, (CTYPE)1, V, wrk1, (CTYPE)0, wrk2);
#else
        FN(gemm_tn)(n_v, n_u, m, (CTYPE)1, V, wrk1, (CTYPE)0, wrk2);
#endif
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
#undef CREAL
