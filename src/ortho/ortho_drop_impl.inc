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
        FN(apply_block_op)(B, V, wrk1, m, n_v);
        BV_norm = FN(nrm2)(m * n_v, wrk1);
    }

    RTYPE U_norm = 0, BU_norm = 0;

    /* Outer loop: orthogonalization iterations */
    for (uint64_t outer = 0; outer < 3; outer++) {
        /* Compute coef = V^H * B * U  (n_v x n_u) stored in wrk2 */
        FN(gram_cross)(V, n_v, U, n_u, m, B, wrk2, n_v, wrk1);

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
            U_norm = FN(nrm2)(m * n_u, U);
            if (NULL == B) {
                BU_norm = U_norm;
            } else {
                FN(apply_block_op)(B, U, wrk1, m, n_u);
                BU_norm = FN(nrm2)(m * n_u, wrk1);
            }
            FN(gram_self)(U, m, n_u, B, wrk2, n_u, wrk1);

            /* Subtract identity: G = G - I */
            for (uint64_t i = 0; i < n_u; i++) {
                wrk2[i + i*n_u] -= (CTYPE)1;
            }

            RTYPE R_norm = FN(nrm2)(n_u * n_u, wrk2);
            RTYPE rerr = R_norm / (BU_norm * U_norm);

            if (rerr < eps_ortho) break;
        }

        /* Check ||V^H * B * U||_F < eps_ortho */
        FN(gram_cross)(V, n_v, U, n_u, m, B, wrk2, n_v, wrk1);

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
