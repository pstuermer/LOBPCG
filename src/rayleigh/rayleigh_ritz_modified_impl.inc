/**
 * @file rayleigh_ritz_modified_impl.inc
 * @brief Type-generic modified Rayleigh-Ritz with condition check and Cx/Cp extraction
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

/* Modified Rayleigh-Ritz with condition number check
 *
 * Inputs:
 *   size: problem dimension
 *   nx: block size (nev)
 *   nconv: number of converged eigenpairs (for soft-locking)
 *   ndrop: number of dropped columns (unused currently)
 *   useOrtho: pointer to orthogonalization flag (input/output)
 *            0 = normal path, 1 = use orthogonalization, 2 = condition check failed
 *   S: search space (size x mult*nx) where mult = 2 or 3
 *
 * Outputs:
 *   Cx: coefficients for X update (mult*nx x nx)
 *   Cp: coefficients for P update (mult*nx x nx)
 *
 *   mult: number of blocks in S (2 for iter 0 [X,W], 3 for iter > 0 [X,P,W])
 *   eigVal: output eigenvalues (array of at least nx reals)
 */
void FN(rayleigh_ritz_modified)(const uint64_t size, const uint64_t nx,
                                const uint64_t mult,
                                const uint64_t nconv, const uint64_t ndrop,
                                uint8_t *useOrtho,
                                CTYPE *restrict S, CTYPE *restrict wrk1,
                                CTYPE *restrict wrk2, CTYPE *restrict wrk3,
                                CTYPE *restrict Cx, CTYPE *restrict Cp,
                                RTYPE *restrict eigVal,
                                LINOP *A, LINOP *B) {
    (void)nconv;  /* TODO: implement soft-locking */
    (void)ndrop;

    const RTYPE tol_skip = 5.0e-3;  /* Condition number threshold */
    const uint64_t sizeSub = mult * nx;

    RTYPE *eigvals_r = xcalloc(sizeSub, sizeof(RTYPE));
    RTYPE *D = xcalloc(sizeSub, sizeof(RTYPE));
    CTYPE *tau = xcalloc(sizeSub, sizeof(CTYPE));

    /* 1. Gram matrix G = S^H * B * S in wrk1 */
    memset(wrk1, 0, sizeSub * sizeSub * sizeof(CTYPE));
    if (NULL == B) {
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
#ifdef CTYPE_IS_COMPLEX
        FN(gemm_hn)(sizeSub, sizeSub, size, (CTYPE)1, S, wrk2, (CTYPE)0, wrk1);
#else
        FN(gemm_tn)(sizeSub, sizeSub, size, (CTYPE)1, S, wrk2, (CTYPE)0, wrk1);
#endif
    }

    /* Fill lower triangle */
    for (uint64_t j = 0; j < sizeSub; j++) {
        for (uint64_t i = j + 1; i < sizeSub; i++) {
#ifdef CTYPE_IS_COMPLEX
            wrk1[i + j*sizeSub] = conj(wrk1[j + i*sizeSub]);
#else
            wrk1[i + j*sizeSub] = wrk1[j + i*sizeSub];
#endif
        }
    }

    /* 2. Diagonal scaling */
    for (uint64_t i = 0; i < sizeSub; i++) {
        RTYPE gii = CABS(wrk1[i + i*sizeSub]);
        D[i] = (gii > 0) ? (RTYPE)1/sqrt(gii) : (RTYPE)1;
    }

    /* 3. Scale: G_scaled = D*G*D */
    for (uint64_t j = 0; j < sizeSub; j++) {
        for (uint64_t i = 0; i < sizeSub; i++) {
            wrk1[i + j*sizeSub] *= D[i] * D[j];
        }
    }

    /* 4. Cholesky factorization */
    if (0 != FN(potrf)(sizeSub, wrk1)) {
        fprintf(stderr, "rayleigh_ritz_modified: Cholesky failed\n");
        *useOrtho = 2;
        safe_free((void**)&eigvals_r);
        safe_free((void**)&D);
        safe_free((void**)&tau);
        return;
    }

    /* 5. Condition number check (if not using orthogonalization) */
    if (0 == *useOrtho) {
        RTYPE rcond;
        if (0 != FN(trcon)('1', sizeSub, wrk1, &rcond)) {
            fprintf(stderr, "rayleigh_ritz_modified: trcon failed\n");
        }
        if (rcond < tol_skip) {
            *useOrtho = 2;  /* Signal condition check failure */
            safe_free((void**)&eigvals_r);
            safe_free((void**)&D);
            safe_free((void**)&tau);
            return;
        }
    }

    /* 6. Compute D_inv_L = D * inv(L^H) in wrk3 */
    memset(wrk3, 0, sizeSub * sizeSub * sizeof(CTYPE));
    for (uint64_t i = 0; i < sizeSub; i++) {
        wrk3[i + i*sizeSub] = (CTYPE)D[i];
    }
#ifdef CTYPE_IS_COMPLEX
    FN(trsm_llh)(sizeSub, sizeSub, (CTYPE)1, wrk1, wrk3);
#else
    FN(trsm_llt)(sizeSub, sizeSub, (CTYPE)1, wrk1, wrk3);
#endif

    /* 7. Compute AS = A * S in wrk2 */
    for (uint64_t j = 0; j < sizeSub; j++) {
        A->matvec(A, &S[j*size], &wrk2[j*size]);
    }

    /* 8. Compute Ap = S^H * AS */
#ifdef CTYPE_IS_COMPLEX
    FN(gemm_hn)(sizeSub, sizeSub, size, (CTYPE)1, S, wrk2, (CTYPE)0, wrk1);
#else
    FN(gemm_tn)(sizeSub, sizeSub, size, (CTYPE)1, S, wrk2, (CTYPE)0, wrk1);
#endif

    /* 9. Transform: Ap_scaled = D_inv_L^H * Ap * D_inv_L */
    FN(gemm_nn)(sizeSub, sizeSub, sizeSub, (CTYPE)1, wrk1, wrk3, (CTYPE)0, Cx);
#ifdef CTYPE_IS_COMPLEX
    FN(gemm_hn)(sizeSub, sizeSub, sizeSub, (CTYPE)1, wrk3, Cx, (CTYPE)0, wrk1);
#else
    FN(gemm_tn)(sizeSub, sizeSub, sizeSub, (CTYPE)1, wrk3, Cx, (CTYPE)0, wrk1);
#endif

    /* Fill lower triangle */
    for (uint64_t j = 0; j < sizeSub; j++) {
        for (uint64_t i = j + 1; i < sizeSub; i++) {
#ifdef CTYPE_IS_COMPLEX
            wrk1[i + j*sizeSub] = conj(wrk1[j + i*sizeSub]);
#else
            wrk1[i + j*sizeSub] = wrk1[j + i*sizeSub];
#endif
        }
    }

    /* 10. Eigensolve */
    if (0 != FN(eig)(sizeSub, wrk1, eigvals_r)) {
        fprintf(stderr, "rayleigh_ritz_modified: eigensolve failed\n");
        safe_free((void**)&eigvals_r);
        safe_free((void**)&D);
        safe_free((void**)&tau);
        return;
    }
    /* wrk1 now contains eigenvectors (column-major) */

    /* 11. Extract Cx = D_inv_L * [first nx columns of wrk1] */
    /* First nx eigenvectors -> Cx */
    memset(Cx, 0, sizeSub * nx * sizeof(CTYPE));
    for (uint64_t j = 0; j < nx; j++) {
        for (uint64_t i = 0; i < sizeSub; i++) {
            Cx[i + j*sizeSub] = wrk1[i + j*sizeSub];
        }
    }
    /* Cx = D_inv_L * Cx */
    FN(gemm_nn)(sizeSub, nx, sizeSub, (CTYPE)1, wrk3, Cx, (CTYPE)0, wrk2);
    memcpy(Cx, wrk2, sizeSub * nx * sizeof(CTYPE));

    /* 12. Extract Z2 = rows [nx : sizeSub] of first nx eigenvectors */
    /*     This is Z2^T (nx x (mult-1)*nx) - rows nx onwards, columns 0 to nx-1 */
    const uint64_t n_remainder = (mult - 1) * nx;
    memset(wrk2, 0, n_remainder * nx * sizeof(CTYPE));
    for (uint64_t i = 0; i < nx; i++) {
        for (uint64_t j = 0; j < n_remainder; j++) {
            wrk2[i + j*nx] = wrk1[nx + j + i*sizeSub];
        }
    }

    /* 13. QR factorization of Z2^T (nx x n_remainder) to get Q_orth */
    /* geqrf: wrk2 contains Z2^T on input, Q factors on output */
    if (0 != FN(geqrf)(nx, n_remainder, wrk2, tau)) {
        fprintf(stderr, "rayleigh_ritz_modified: geqrf failed\n");
        safe_free((void**)&eigvals_r);
        safe_free((void**)&D);
        safe_free((void**)&tau);
        return;
    }

    /* ungqr/orgqr: generate Q from Householder reflectors (nx x nx submatrix) */
    uint64_t k = MIN(nx, n_remainder);
#ifdef CTYPE_IS_COMPLEX
    if (0 != FN(ungqr)(nx, nx, k, wrk2, tau)) {
#else
    if (0 != FN(orgqr)(nx, nx, k, wrk2, tau)) {
#endif
        fprintf(stderr, "rayleigh_ritz_modified: ungqr/orgqr failed\n");
        safe_free((void**)&eigvals_r);
        safe_free((void**)&D);
        safe_free((void**)&tau);
        return;
    }
    /* wrk2 now contains Q_orth (nx x nx, only first nx columns) */

    /* 14. Extract [Z2_full] = rows [nx : sizeSub] of eigenvectors nx onwards */
    /*     Z2_full is (sizeSub-nx) rows x nx columns, from eigenvectors nx onwards */
    memset(Cp, 0, sizeSub * n_remainder * sizeof(CTYPE));
    for (uint64_t j = 0; j < n_remainder; j++) {
        for (uint64_t i = 0; i < sizeSub; i++) {
            Cp[i + j*sizeSub] = wrk1[i + (nx + j)*sizeSub];
        }
    }

    /* 15. Cp_tmp = D_inv_L * Cp */
    FN(gemm_nn)(sizeSub, n_remainder, sizeSub, (CTYPE)1, wrk3, Cp, (CTYPE)0, wrk1);

    /* 16. Cp_final = Cp_tmp * Q_orth */
    /*     wrk1 is (sizeSub x n_remainder), wrk2 is (nx x nx) */
    /*     We need (sizeSub x nx) = (sizeSub x n_remainder) * (n_remainder x nx) */
    /*     But Q_orth from ungqr is (nx x nx), we need to use first nx cols */
    /*     Actually, we computed Q for Z2^T which is (nx x n_remainder) */
    /*     Q is orthonormal basis of null(Z1) which is nx x nx */
    memset(Cp, 0, sizeSub * nx * sizeof(CTYPE));
    FN(gemm_nn)(sizeSub, nx, nx, (CTYPE)1, wrk1, wrk2, (CTYPE)0, Cp);

    /* Copy eigenvalues to output */
    for (uint64_t i = 0; i < nx; i++) {
        eigVal[i] = eigvals_r[i];
    }

    safe_free((void**)&eigvals_r);
    safe_free((void**)&D);
    safe_free((void**)&tau);
}

#undef CONCAT2
#undef CONCAT
#undef FN
#undef MIN
#undef CABS
