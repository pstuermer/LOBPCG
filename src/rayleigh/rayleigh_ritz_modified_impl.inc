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
    // TODO: this is not correct, should be mult*nx - ndrop - nconv
    // to get the correct matrix size
    const uint64_t sizeSub = mult * nx;
    const uint64_t n_remainder = (mult - 1) * nx;

    RTYPE *eigvals_r = xcalloc(sizeSub, sizeof(RTYPE));
    CTYPE *tau = xcalloc(sizeSub, sizeof(CTYPE));

    /* ================================================================
     * Ortho branch: useOrtho==1
     * Direct eigensolve on S^H * A * S (no Gram/Cholesky/D_inv_R)
     * ================================================================ */
    if (*useOrtho) {
        *useOrtho = 1;

        /* Compute AS = A * S in wrk2 */
        for (uint64_t j = 0; j < sizeSub; j++) {
            A->matvec(A, &S[j*size], &wrk2[j*size]);
        }

        /* Form wrk1 = S^H * AS */
        memset(wrk1, 0, sizeSub * sizeSub * sizeof(CTYPE));
#ifdef CTYPE_IS_COMPLEX
        FN(gemm_hn)(sizeSub, sizeSub, size, (CTYPE)1, S, wrk2, (CTYPE)0, wrk1);
#else
        FN(gemm_tn)(sizeSub, sizeSub, size, (CTYPE)1, S, wrk2, (CTYPE)0, wrk1);
#endif

        /* Eigensolve on full sizeSub x sizeSub matrix */
        if (0 != FN(eig)(sizeSub, wrk1, eigvals_r)) {
            fprintf(stderr, "rayleigh_ritz_modified(ortho): eigensolve failed\n");
            safe_free((void**)&eigvals_r);
            safe_free((void**)&tau);
            return;
        }
        /* wrk1 now contains eigenvectors (column-major) */

        /* Extract Z2 (n_remainder x nx) from rows [nx:sizeSub] of first nx eigvecs */
        memset(wrk2, 0, n_remainder * nx * sizeof(CTYPE));
        for (uint64_t i = 0; i < nx; i++) {
            for (uint64_t j = 0; j < n_remainder; j++) {
                wrk2[j + i*n_remainder] = wrk1[(nx + j) + i*sizeSub];
            }
        }

        /* QR factorization of Z2 (n_remainder x nx) — tall matrix */
        if (0 != FN(geqrf)(n_remainder, nx, wrk2, tau)) {
            fprintf(stderr, "rayleigh_ritz_modified(ortho): geqrf failed\n");
            safe_free((void**)&eigvals_r);
            safe_free((void**)&tau);
            return;
        }

        /* Generate Q_orth (n_remainder x nx) */
#ifdef CTYPE_IS_COMPLEX
        if (0 != FN(ungqr)(n_remainder, nx, nx, wrk2, tau)) {
#else
        if (0 != FN(orgqr)(n_remainder, nx, nx, wrk2, tau)) {
#endif
            fprintf(stderr, "rayleigh_ritz_modified(ortho): ungqr/orgqr failed\n");
            safe_free((void**)&eigvals_r);
            safe_free((void**)&tau);
            return;
        }
        /* wrk2 now contains Q_orth (n_remainder x nx) */

        /* Cx = first nx columns of eigenvectors (no D_inv_R transform) */
        memset(Cx, 0, sizeSub * nx * sizeof(CTYPE));
        for (uint64_t j = 0; j < nx; j++) {
            for (uint64_t i = 0; i < sizeSub; i++) {
                Cx[i + j*sizeSub] = wrk1[i + j*sizeSub];
            }
        }

        /* Extract full eigvecs [nx:end, nx:end] for Cp computation
         * wrk3 = all sizeSub rows of eigvecs nx..sizeSub (n_remainder cols) */
        memset(wrk3, 0, sizeSub * n_remainder * sizeof(CTYPE));
        for (uint64_t j = 0; j < n_remainder; j++) {
            for (uint64_t i = 0; i < sizeSub; i++) {
                wrk3[i + j*sizeSub] = wrk1[i + (nx + j)*sizeSub];
            }
        }

        /* Cp = wrk3 * Q_orth  (sizeSub x n_remainder) * (n_remainder x nx) -> (sizeSub x nx) */
        memset(Cp, 0, sizeSub * nx * sizeof(CTYPE));
        FN(gemm_nn)(sizeSub, nx, n_remainder, (CTYPE)1, wrk3, wrk2, (CTYPE)0, Cp);

        /* Copy eigenvalues */
        for (uint64_t i = 0; i < nx; i++) {
            eigVal[i] = eigvals_r[i];
        }

        safe_free((void**)&eigvals_r);
        safe_free((void**)&tau);
        return;
    }

    /* ================================================================
     * Cholesky branch: useOrtho==0
     * Gram matrix + Cholesky + condition check + D_inv_R transform
     * ================================================================ */
    RTYPE *D = xcalloc(sizeSub, sizeof(RTYPE));

    /* 1. Gram matrix G = S^H * B * S in wrk1 (upper triangle) */
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

    /* 4. Cholesky factorization: wrk1 = R^H * R (R in upper triangle) */
    if (0 != FN(potrf)(sizeSub, wrk1)) {
        fprintf(stderr, "rayleigh_ritz_modified: Cholesky failed\n");
        *useOrtho = 2;
        safe_free((void**)&eigvals_r);
        safe_free((void**)&D);
        safe_free((void**)&tau);
        return;
    }

    /* 5. Condition number check */
    {
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

    /* 6. Compute D_inv_R = D * inv(R) in wrk3
     *    Set wrk3 = D, solve wrk3 * R = D  =>  wrk3 = D * inv(R) */
    memset(wrk3, 0, sizeSub * sizeSub * sizeof(CTYPE));
    for (uint64_t i = 0; i < sizeSub; i++) {
        wrk3[i + i*sizeSub] = (CTYPE)D[i];
    }
    FN(trsm_run)(sizeSub, sizeSub, (CTYPE)1, wrk1, wrk3);

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

    /* 9. Transform: Ap_scaled = D_inv_R^H * Ap * D_inv_R */
    FN(gemm_nn)(sizeSub, sizeSub, sizeSub, (CTYPE)1, wrk1, wrk3, (CTYPE)0, Cx);
#ifdef CTYPE_IS_COMPLEX
    FN(gemm_hn)(sizeSub, sizeSub, sizeSub, (CTYPE)1, wrk3, Cx, (CTYPE)0, wrk1);
#else
    FN(gemm_tn)(sizeSub, sizeSub, sizeSub, (CTYPE)1, wrk3, Cx, (CTYPE)0, wrk1);
#endif

    /* 10. Eigensolve (reads upper triangle) */
    if (0 != FN(eig)(sizeSub, wrk1, eigvals_r)) {
        fprintf(stderr, "rayleigh_ritz_modified: eigensolve failed\n");
        safe_free((void**)&eigvals_r);
        safe_free((void**)&D);
        safe_free((void**)&tau);
        return;
    }
    /* wrk1 now contains eigenvectors (column-major) */

    /* 11. Extract Cx = D_inv_R * [first nx columns of wrk1] */
    memset(Cx, 0, sizeSub * nx * sizeof(CTYPE));
    for (uint64_t j = 0; j < nx; j++) {
        for (uint64_t i = 0; i < sizeSub; i++) {
            Cx[i + j*sizeSub] = wrk1[i + j*sizeSub];
        }
    }
    /* Cx = D_inv_R * Cx */
    FN(gemm_nn)(sizeSub, nx, sizeSub, (CTYPE)1, wrk3, Cx, (CTYPE)0, wrk2);
    memcpy(Cx, wrk2, sizeSub * nx * sizeof(CTYPE));

    /* 12. Extract Z2 (n_remainder x nx) — rows [nx:sizeSub] of first nx eigenvectors */
    memset(wrk2, 0, n_remainder * nx * sizeof(CTYPE));
    for (uint64_t i = 0; i < nx; i++) {
        for (uint64_t j = 0; j < n_remainder; j++) {
            wrk2[j + i*n_remainder] = wrk1[(nx + j) + i*sizeSub];
        }
    }

    /* 13. QR factorization of Z2 (n_remainder x nx) — tall matrix */
    if (0 != FN(geqrf)(n_remainder, nx, wrk2, tau)) {
        fprintf(stderr, "rayleigh_ritz_modified: geqrf failed\n");
        safe_free((void**)&eigvals_r);
        safe_free((void**)&D);
        safe_free((void**)&tau);
        return;
    }

    /* ungqr/orgqr: generate Q from Householder reflectors (n_remainder x nx) */
#ifdef CTYPE_IS_COMPLEX
    if (0 != FN(ungqr)(n_remainder, nx, nx, wrk2, tau)) {
#else
    if (0 != FN(orgqr)(n_remainder, nx, nx, wrk2, tau)) {
#endif
        fprintf(stderr, "rayleigh_ritz_modified: ungqr/orgqr failed\n");
        safe_free((void**)&eigvals_r);
        safe_free((void**)&D);
        safe_free((void**)&tau);
        return;
    }
    /* wrk2 now contains Q_orth (n_remainder x nx) */

    /* 14. Extract [Z2_full] = rows [nx : sizeSub] of eigenvectors nx onwards */
    memset(Cp, 0, sizeSub * n_remainder * sizeof(CTYPE));
    for (uint64_t j = 0; j < n_remainder; j++) {
        for (uint64_t i = 0; i < sizeSub; i++) {
            Cp[i + j*sizeSub] = wrk1[i + (nx + j)*sizeSub];
        }
    }

    /* 15. Cp_tmp = D_inv_R * Cp */
    FN(gemm_nn)(sizeSub, n_remainder, sizeSub, (CTYPE)1, wrk3, Cp, (CTYPE)0, wrk1);

    /* 16. Cp_final = Cp_tmp * Q_orth  (sizeSub x n_remainder) * (n_remainder x nx) */
    memset(Cp, 0, sizeSub * nx * sizeof(CTYPE));
    FN(gemm_nn)(sizeSub, nx, n_remainder, (CTYPE)1, wrk1, wrk2, (CTYPE)0, Cp);

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
