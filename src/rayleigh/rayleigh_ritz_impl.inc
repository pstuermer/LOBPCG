/**
 * @file rayleigh_ritz_impl.inc
 * @brief Type-generic standard Rayleigh-Ritz procedure
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

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifdef CTYPE_IS_COMPLEX
  #define CABS(x) cabs(x)
#else
  #define CABS(x) fabs(x)
#endif

/* Standard Rayleigh-Ritz procedure
 *
 * Algorithm:
 * 1. Compute Gram matrix G = S^H * B * S
 * 2. Diagonal scaling D_ii = 1/sqrt(|G_ii|)
 * 3. Compute scaled Gram: G_scaled = D * G * D
 * 4. Cholesky: G_scaled = L * L^H
 * 5. Compute D_inv_L = D * inv(L^H)
 * 6. Compute AS = A * S
 * 7. Compute Ap = S^H * AS
 * 8. Transform: Ap_scaled = D_inv_L^H * Ap * D_inv_L
 * 9. Eigensolve: Ap_scaled * V = V * Lambda
 * 10. Back-transform: Cx = D_inv_L * V
 * 11. Store eigenvalues in eigVal
 */
void FN(rayleigh_ritz)(const uint64_t size, const uint64_t sizeSub,
                       CTYPE *restrict S, CTYPE *restrict Cx, CTYPE *restrict eigVal,
                       CTYPE *restrict wrk1, CTYPE *restrict wrk2, CTYPE *restrict wrk3,
                       LINOP *A, LINOP *B) {

    RTYPE *eigvals_r = xcalloc(sizeSub, sizeof(RTYPE));
    RTYPE *D = xcalloc(sizeSub, sizeof(RTYPE));

    /* 1. Gram matrix G = S^H * B * S in wrk1 (sizeSub x sizeSub) */
    memset(wrk1, 0, sizeSub * sizeSub * sizeof(CTYPE));
    if (NULL == B) {
        /* G = S^H * S via herk/syrk */
#ifdef CTYPE_IS_COMPLEX
        FN(herk)(size, sizeSub, (RTYPE)1, S, (RTYPE)0, wrk1);
#else
        FN(syrk)(size, sizeSub, (RTYPE)1, S, (RTYPE)0, wrk1);
#endif
    } else {
        /* BS = B*S in wrk2 */
        for (uint64_t j = 0; j < sizeSub; j++) {
            B->matvec(B, &S[j*size], &wrk2[j*size]);
        }
        /* G = S^H * BS */
#ifdef CTYPE_IS_COMPLEX
        FN(gemm_hn)(sizeSub, sizeSub, size, (CTYPE)1, S, wrk2, (CTYPE)0, wrk1);
#else
        FN(gemm_tn)(sizeSub, sizeSub, size, (CTYPE)1, S, wrk2, (CTYPE)0, wrk1);
#endif
    }

    /* Fill lower triangle from upper for symmetric/Hermitian matrix */
    for (uint64_t j = 0; j < sizeSub; j++) {
        for (uint64_t i = j + 1; i < sizeSub; i++) {
#ifdef CTYPE_IS_COMPLEX
            wrk1[i + j*sizeSub] = conj(wrk1[j + i*sizeSub]);
#else
            wrk1[i + j*sizeSub] = wrk1[j + i*sizeSub];
#endif
        }
    }

    /* 2. Diagonal scaling D_ii = 1/sqrt(|G_ii|) */
    for (uint64_t i = 0; i < sizeSub; i++) {
        RTYPE gii = CABS(wrk1[i + i*sizeSub]);
        D[i] = (gii > 0) ? (RTYPE)1/sqrt(gii) : (RTYPE)1;
    }

    /* 3. Scale: G_scaled = D*G*D in wrk1 */
    for (uint64_t j = 0; j < sizeSub; j++) {
        for (uint64_t i = 0; i < sizeSub; i++) {
            wrk1[i + j*sizeSub] *= D[i] * D[j];
        }
    }

    /* 4. Cholesky factorization: wrk1 = L * L^H (lower triangular) */
    if (0 != FN(potrf)(sizeSub, wrk1)) {
        fprintf(stderr, "rayleigh_ritz: Cholesky failed\n");
        safe_free((void**)&eigvals_r);
        safe_free((void**)&D);
        return;
    }

    /* 5. Compute D_inv_L = D * inv(L^H) in wrk3 */
    /* First, compute inv(L^H) via trsm: wrk3 = D, solve L^H * X = D */
    memset(wrk3, 0, sizeSub * sizeSub * sizeof(CTYPE));
    for (uint64_t i = 0; i < sizeSub; i++) {
        wrk3[i + i*sizeSub] = (CTYPE)D[i];
    }
#ifdef CTYPE_IS_COMPLEX
    FN(trsm_llh)(sizeSub, sizeSub, (CTYPE)1, wrk1, wrk3);
#else
    FN(trsm_llt)(sizeSub, sizeSub, (CTYPE)1, wrk1, wrk3);
#endif

    /* 6. Compute AS = A * S in wrk2 */
    for (uint64_t j = 0; j < sizeSub; j++) {
        A->matvec(A, &S[j*size], &wrk2[j*size]);
    }

    /* 7. Compute Ap = S^H * AS in wrk1 */
#ifdef CTYPE_IS_COMPLEX
    FN(gemm_hn)(sizeSub, sizeSub, size, (CTYPE)1, S, wrk2, (CTYPE)0, wrk1);
#else
    FN(gemm_tn)(sizeSub, sizeSub, size, (CTYPE)1, S, wrk2, (CTYPE)0, wrk1);
#endif

    /* 8. Transform: Ap_scaled = D_inv_L^H * Ap * D_inv_L */
    /* First: tmp = Ap * D_inv_L in Cx (use as temp) */
    FN(gemm_nn)(sizeSub, sizeSub, sizeSub, (CTYPE)1, wrk1, wrk3, (CTYPE)0, Cx);
    /* Then: Ap_scaled = D_inv_L^H * tmp in wrk1 */
#ifdef CTYPE_IS_COMPLEX
    FN(gemm_hn)(sizeSub, sizeSub, sizeSub, (CTYPE)1, wrk3, Cx, (CTYPE)0, wrk1);
#else
    FN(gemm_tn)(sizeSub, sizeSub, sizeSub, (CTYPE)1, wrk3, Cx, (CTYPE)0, wrk1);
#endif

    /* Fill lower triangle for eigensolve */
    for (uint64_t j = 0; j < sizeSub; j++) {
        for (uint64_t i = j + 1; i < sizeSub; i++) {
#ifdef CTYPE_IS_COMPLEX
            wrk1[i + j*sizeSub] = conj(wrk1[j + i*sizeSub]);
#else
            wrk1[i + j*sizeSub] = wrk1[j + i*sizeSub];
#endif
        }
    }

    /* 9. Eigensolve: wrk1 contains eigenvectors on return */
    if (0 != FN(eig)(sizeSub, wrk1, eigvals_r)) {
        fprintf(stderr, "rayleigh_ritz: eigensolve failed\n");
        safe_free((void**)&eigvals_r);
        safe_free((void**)&D);
        return;
    }

    /* 10. Back-transform: Cx = D_inv_L * V */
    FN(gemm_nn)(sizeSub, sizeSub, sizeSub, (CTYPE)1, wrk3, wrk1, (CTYPE)0, Cx);

    /* 11. Store eigenvalues (real eigenvalues in complex array for consistency) */
    for (uint64_t i = 0; i < sizeSub; i++) {
#ifdef CTYPE_IS_COMPLEX
        eigVal[i] = eigvals_r[i] + 0*I;
#else
        eigVal[i] = eigvals_r[i];
#endif
    }

    safe_free((void**)&eigvals_r);
    safe_free((void**)&D);
}

#undef CONCAT2
#undef CONCAT
#undef FN
#undef MIN
#undef CABS
