/**
 * @file ortho_drop_impl.inc
 * @brief Type-generic B-orthogonalization of U against V (positive definite B)
 *
 * Macros required: PREFIX, CTYPE, RTYPE, LINOP
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
  #define CONJ(x) conj(x)
#else
  #define CABS(x) fabs(x)
  #define CONJ(x) (x)
#endif

/* B-orthogonalize U against V with positive definite B
 *
 * Algorithm:
 * 1. Outer loop (max 3 iterations):
 *    a. Project: U = U - V * (V^H * B * U)
 *    b. Inner loop (max 3 iterations):
 *       - svqb to B-orthonormalize U
 *       - Check ||U^H * B * U - I||_F < eps_ortho
 *    c. Check ||V^H * B * U||_F < eps_ortho
 *
 * Returns: number of columns retained in U
 */
uint64_t FN(ortho_drop)(const uint64_t m, const uint64_t n_u, const uint64_t n_v,
                        const RTYPE eps_ortho, const RTYPE eps_drop,
                        CTYPE *restrict U, CTYPE *restrict V,
                        CTYPE *restrict wrk1, CTYPE *restrict wrk2, CTYPE *restrict wrk3,
                        LINOP *B) {
    if (0 == n_u || 0 == n_v) return n_u;

    /* Check dimension consistency */
    if (m < n_u + n_v) {
        fprintf(stderr, "ortho_drop: overdetermined shape m=%lu < n_u+n_v=%lu+%lu\n",
                (unsigned long)m, (unsigned long)n_u, (unsigned long)n_v);
        return n_u;
    }

    /* Compute ||B*V||_F for relative tolerance */
    RTYPE BV_norm = 0;
    if (NULL == B) {
        BV_norm = FN(nrm2)(m * n_v, V);
    } else {
        /* BV = B*V in wrk1 */
        for (uint64_t j = 0; j < n_v; j++) {
            B->matvec(B, &V[j*m], &wrk1[j*m]);
        }
        BV_norm = FN(nrm2)(m * n_v, wrk1);
    }

    RTYPE U_norm = 0, BU_norm = 0;

    /* Outer loop: orthogonalization iterations */
    for (uint64_t outer = 0; outer < 3; outer++) {
        /* Compute coef = V^H * B * U  (n_v x n_u) stored in wrk2 */
        if (NULL == B) {
            /* coef = V^H * U */
#ifdef CTYPE_IS_COMPLEX
            FN(gemm_hn)(n_v, n_u, m, (CTYPE)1, V, U, (CTYPE)0, wrk2);
#else
            FN(gemm_tn)(n_v, n_u, m, (CTYPE)1, V, U, (CTYPE)0, wrk2);
#endif
        } else {
            /* BU = B*U in wrk1 */
            for (uint64_t j = 0; j < n_u; j++) {
                B->matvec(B, &U[j*m], &wrk1[j*m]);
            }
            /* coef = V^H * BU */
#ifdef CTYPE_IS_COMPLEX
            FN(gemm_hn)(n_v, n_u, m, (CTYPE)1, V, wrk1, (CTYPE)0, wrk2);
#else
            FN(gemm_tn)(n_v, n_u, m, (CTYPE)1, V, wrk1, (CTYPE)0, wrk2);
#endif
        }

        /* U = U - V * coef */
        FN(gemm_nn)(m, n_u, n_v, (CTYPE)(-1), V, wrk2, (CTYPE)1, U);

        /* Inner loop: svqb iterations */
        for (uint64_t inner = 0; inner < 3; inner++) {
            /* svqb to B-orthonormalize U */
            uint64_t n_ret = FN(svqb)(m, n_u, eps_drop, 'n', U, wrk1, wrk2, wrk3, B);

            if (n_ret != n_u) {
                fprintf(stderr, "ortho_drop: svqb dropped columns (%lu -> %lu)\n",
                        (unsigned long)n_u, (unsigned long)n_ret);
                return n_ret;
            }

            /* Check ||U^H * B * U - I||_F < eps_ortho */
            if (NULL == B) {
                U_norm = FN(nrm2)(m * n_u, U);
                BU_norm = U_norm;
                /* G = U^H * U */
#ifdef CTYPE_IS_COMPLEX
                FN(gemm_hn)(n_u, n_u, m, (CTYPE)1, U, U, (CTYPE)0, wrk2);
#else
                FN(gemm_tn)(n_u, n_u, m, (CTYPE)1, U, U, (CTYPE)0, wrk2);
#endif
            } else {
                /* BU = B*U */
                for (uint64_t j = 0; j < n_u; j++) {
                    B->matvec(B, &U[j*m], &wrk1[j*m]);
                }
                U_norm = FN(nrm2)(m * n_u, U);
                BU_norm = FN(nrm2)(m * n_u, wrk1);
                /* G = U^H * BU */
#ifdef CTYPE_IS_COMPLEX
                FN(gemm_hn)(n_u, n_u, m, (CTYPE)1, U, wrk1, (CTYPE)0, wrk2);
#else
                FN(gemm_tn)(n_u, n_u, m, (CTYPE)1, U, wrk1, (CTYPE)0, wrk2);
#endif
            }

            /* Subtract identity: G = G - I */
            for (uint64_t i = 0; i < n_u; i++) {
                wrk2[i + i*n_u] -= (CTYPE)1;
            }

            RTYPE R_norm = FN(nrm2)(n_u * n_u, wrk2);
            RTYPE rerr = R_norm / (BU_norm * U_norm);

            if (rerr < eps_ortho) break;
        }

        /* Check ||V^H * B * U||_F < eps_ortho */
        if (NULL == B) {
            /* coef = V^H * U (already computed, in wrk2) */
#ifdef CTYPE_IS_COMPLEX
            FN(gemm_hn)(n_v, n_u, m, (CTYPE)1, V, U, (CTYPE)0, wrk2);
#else
            FN(gemm_tn)(n_v, n_u, m, (CTYPE)1, V, U, (CTYPE)0, wrk2);
#endif
        } else {
            /* BU already in wrk1 from inner loop */
            /* coef = V^H * BU */
#ifdef CTYPE_IS_COMPLEX
            FN(gemm_hn)(n_v, n_u, m, (CTYPE)1, V, wrk1, (CTYPE)0, wrk2);
#else
            FN(gemm_tn)(n_v, n_u, m, (CTYPE)1, V, wrk1, (CTYPE)0, wrk2);
#endif
        }

        RTYPE VU_norm = FN(nrm2)(n_v * n_u, wrk2);
        U_norm = FN(nrm2)(m * n_u, U);
        RTYPE rerr = VU_norm / (BV_norm * U_norm);

        if (rerr < eps_ortho) break;
    }

    return n_u;
}

#undef CONCAT2
#undef CONCAT
#undef FN
#undef MAX
#undef CABS
#undef CONJ
