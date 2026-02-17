/**
 * @file indefinite_rr_modified_impl.inc
 * @brief Type-generic modified indefinite Rayleigh-Ritz with Cx/Cp extraction
 *
 * Same as indefinite_rr but for the iterative LOBPCG loop:
 * - S has mult*nx columns (mult=2 for [X,W], mult=3 for [X,P,W])
 * - Extracts Cx (first nx eigenvectors) and Cp (via QR of remainder)
 * - Returns RTYPE eigenvalues (real part)
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
  #define CREAL(x) creal(x)
#else
  #define CABS(x) fabs(x)
  #define CREAL(x) (x)
#endif

/* Bubble sort eigenvalues by signature (modified variant uses RTYPE eigvals) */
static void FN(bubble_sort_sig_mod)(RTYPE *eigVal, int8_t *signature,
                                     uint64_t *indices, uint64_t n) {
    for (uint64_t i = 0; i < n; i++)
        indices[i] = i;

    for (uint64_t i = 0; i < n; i++) {
        int swapped = 0;
        for (uint64_t j = 0; j + 1 < n - i; j++) {
            int8_t siga = signature[j];
            int8_t sigb = signature[j + 1];
            RTYPE ra = eigVal[j];
            RTYPE rb = eigVal[j + 1];
            int should_swap = 0;

            if (siga > 0 && sigb > 0) {
                should_swap = (ra > rb);
            } else if (siga < 0 && sigb < 0) {
                should_swap = (ra < rb);
            } else if (siga < 0 && sigb > 0) {
                should_swap = 1;
            }

            if (should_swap) {
                RTYPE tmp_e = eigVal[j];
                eigVal[j] = eigVal[j + 1];
                eigVal[j + 1] = tmp_e;

                int8_t tmp_s = signature[j];
                signature[j] = signature[j + 1];
                signature[j + 1] = tmp_s;

                uint64_t tmp_i = indices[j];
                indices[j] = indices[j + 1];
                indices[j + 1] = tmp_i;

                swapped = 1;
            }
        }
        if (!swapped) break;
    }
}

/* Modified indefinite Rayleigh-Ritz
 *
 * Inputs:
 *   size:      problem dimension
 *   nx:        block size (nev)
 *   mult:      block multiplier (2 or 3), sizeSub = mult*nx
 *   nconv:     converged count (unused, for soft-locking)
 *   ndrop:     dropped columns (unused)
 *   S:         search space (size x sizeSub)
 *   A, B:      operators (B indefinite, must not be NULL)
 *
 * Outputs:
 *   Cx:        coefficients for X update (sizeSub x nx)
 *   Cp:        coefficients for P update (sizeSub x nx)
 *   eigVal:    first nx eigenvalues (RTYPE, real part)
 *   signature: +1/-1 per eigenpair (length sizeSub, sorted)
 *
 * Workspace:
 *   wrk1: sizeSub x sizeSub
 *   wrk2: size x sizeSub
 *   wrk3: sizeSub x sizeSub
 *   wrk4: sizeSub x sizeSub
 */
void FN(indefinite_rayleigh_ritz_modified)(
    const uint64_t size, const uint64_t nx,
    const uint64_t mult,
    const uint64_t nconv, const uint64_t ndrop,
    CTYPE *restrict S, CTYPE *restrict wrk1,
    CTYPE *restrict wrk2, CTYPE *restrict wrk3, CTYPE *restrict wrk4,
    CTYPE *restrict Cx, CTYPE *restrict Cp,
    RTYPE *restrict eigVal, int8_t *restrict signature,
    LINOP *A, LINOP *B) {

    (void)nconv;
    (void)ndrop;

    const uint64_t sizeSub = mult * nx;
    const uint64_t ss = sizeSub * sizeSub;

    /* Internal allocations */
    RTYPE *eigvals_all = xcalloc(sizeSub, sizeof(RTYPE));
    int8_t *sig_all = xcalloc(sizeSub, sizeof(int8_t));
    uint64_t *indices = xcalloc(sizeSub, sizeof(uint64_t));
    CTYPE *tau = xcalloc(sizeSub, sizeof(CTYPE));
    CTYPE *VR = xcalloc(ss, sizeof(CTYPE)); /* GGEV right eigenvectors */
#ifdef CTYPE_IS_COMPLEX
    CTYPE *alpha = xcalloc(sizeSub, sizeof(CTYPE));
    CTYPE *beta  = xcalloc(sizeSub, sizeof(CTYPE));
#else
    RTYPE *alphar = xcalloc(sizeSub, sizeof(RTYPE));
    RTYPE *alphai = xcalloc(sizeSub, sizeof(RTYPE));
    RTYPE *beta   = xcalloc(sizeSub, sizeof(RTYPE));
#endif

    /* 1. GA = S^H * A * S in wrk1 */
    for (uint64_t j = 0; j < sizeSub; j++)
        A->matvec(A, &S[j * size], &wrk2[j * size]);
#ifdef CTYPE_IS_COMPLEX
    FN(gemm_hn)(sizeSub, sizeSub, size, (CTYPE)1, S, wrk2, (CTYPE)0, wrk1);
#else
    FN(gemm_tn)(sizeSub, sizeSub, size, (CTYPE)1, S, wrk2, (CTYPE)0, wrk1);
#endif

    /* 2. GB = S^H * B * S in wrk3 */
    for (uint64_t j = 0; j < sizeSub; j++)
        B->matvec(B, &S[j * size], &wrk2[j * size]);
#ifdef CTYPE_IS_COMPLEX
    FN(gemm_hn)(sizeSub, sizeSub, size, (CTYPE)1, S, wrk2, (CTYPE)0, wrk3);
#else
    FN(gemm_tn)(sizeSub, sizeSub, size, (CTYPE)1, S, wrk2, (CTYPE)0, wrk3);
#endif

    /* 3. Copy GA -> wrk4, GB -> wrk2 (GGEV destroys both) */
    memcpy(wrk4, wrk1, ss * sizeof(CTYPE));
    memcpy(wrk2, wrk3, ss * sizeof(CTYPE));

    /* 4. GGEV */
#ifdef CTYPE_IS_COMPLEX
    if (0 != FN(ggev)(sizeSub, wrk4, wrk2, alpha, beta, VR)) {
        fprintf(stderr, "indefinite_rayleigh_ritz_modified: GGEV failed\n");
        goto cleanup;
    }
#else
    if (0 != FN(ggev)(sizeSub, wrk4, wrk2, alphar, alphai, beta, VR)) {
        fprintf(stderr, "indefinite_rayleigh_ritz_modified: GGEV failed\n");
        goto cleanup;
    }
#endif

    /* 5. Compute eigenvalues */
    for (uint64_t i = 0; i < sizeSub; i++) {
#ifdef CTYPE_IS_COMPLEX
        CTYPE ev = alpha[i] / beta[i];
        eigvals_all[i] = CREAL(ev);
#else
        if (CABS(beta[i]) > (RTYPE)1e-30)
            eigvals_all[i] = alphar[i] / beta[i];
        else
            eigvals_all[i] = (alphar[i] >= 0) ? (RTYPE)1e30 : (RTYPE)-1e30;
#endif
    }

    /* 6. B-normalize eigenvectors: V^H * GB * V
     *    wrk3 still holds GB, VR holds eigenvectors */
    FN(gemm_nn)(sizeSub, sizeSub, sizeSub, (CTYPE)1, wrk3, VR, (CTYPE)0, wrk1);
#ifdef CTYPE_IS_COMPLEX
    FN(gemm_hn)(sizeSub, sizeSub, sizeSub, (CTYPE)1, VR, wrk1, (CTYPE)0, wrk4);
#else
    FN(gemm_tn)(sizeSub, sizeSub, sizeSub, (CTYPE)1, VR, wrk1, (CTYPE)0, wrk4);
#endif

    for (uint64_t i = 0; i < sizeSub; i++) {
        CTYPE diag_val = wrk4[i + i * sizeSub];
        RTYPE abs_diag = CABS(diag_val);

        if (abs_diag > (RTYPE)1e-30) {
            RTYPE scale = (RTYPE)1 / sqrt(abs_diag);
            for (uint64_t k = 0; k < sizeSub; k++)
                VR[k + i * sizeSub] *= (CTYPE)scale;
        }

        sig_all[i] = (CREAL(diag_val) >= 0) ? 1 : -1;
    }

    /* 7. Sort by signature */
    FN(bubble_sort_sig_mod)(eigvals_all, sig_all, indices, sizeSub);

    /* 8. Reorder VR columns according to sort permutation -> wrk1 as temp */
    memcpy(wrk1, VR, ss * sizeof(CTYPE));
    for (uint64_t j = 0; j < sizeSub; j++) {
        memcpy(&VR[j * sizeSub], &wrk1[indices[j] * sizeSub],
               sizeSub * sizeof(CTYPE));
    }

    /* 9. Extract first nx eigenvalues and signature */
    for (uint64_t i = 0; i < nx; i++)
        eigVal[i] = eigvals_all[i];
    for (uint64_t i = 0; i < sizeSub; i++)
        signature[i] = sig_all[i];

    /* 10. Extract Cx = first nx columns of sorted VR */
    memset(Cx, 0, sizeSub * nx * sizeof(CTYPE));
    for (uint64_t j = 0; j < nx; j++) {
        for (uint64_t i = 0; i < sizeSub; i++) {
            Cx[i + j * sizeSub] = VR[i + j * sizeSub];
        }
    }

    /* 11. Extract Cp via QR of Z2 (same pattern as rayleigh_ritz_modified)
     *     Z2 = rows [nx : sizeSub] of first nx eigenvectors, transposed */
    const uint64_t n_remainder = (mult - 1) * nx;
    memset(wrk2, 0, n_remainder * nx * sizeof(CTYPE));
    for (uint64_t i = 0; i < nx; i++) {
        for (uint64_t j = 0; j < n_remainder; j++) {
            wrk2[i + j * nx] = VR[nx + j + i * sizeSub];
        }
    }

    /* QR factorization of Z2^T (nx x n_remainder) */
    if (0 != FN(geqrf)(nx, n_remainder, wrk2, tau)) {
        fprintf(stderr, "indefinite_rayleigh_ritz_modified: geqrf failed\n");
        goto cleanup;
    }

    uint64_t k = MIN(nx, n_remainder);
#ifdef CTYPE_IS_COMPLEX
    if (0 != FN(ungqr)(nx, nx, k, wrk2, tau)) {
#else
    if (0 != FN(orgqr)(nx, nx, k, wrk2, tau)) {
#endif
        fprintf(stderr, "indefinite_rayleigh_ritz_modified: ungqr/orgqr failed\n");
        goto cleanup;
    }

    /* Extract columns nx onwards from sorted VR -> Cp_tmp */
    memset(Cp, 0, sizeSub * n_remainder * sizeof(CTYPE));
    for (uint64_t j = 0; j < n_remainder; j++) {
        for (uint64_t i = 0; i < sizeSub; i++) {
            Cp[i + j * sizeSub] = VR[i + (nx + j) * sizeSub];
        }
    }

    /* Cp_final = Cp_tmp * Q_orth
     * wrk1 is used as temp: Cp is (sizeSub x n_remainder), wrk2 is (nx x nx)
     * Result: Cp = (sizeSub x nx) */
    memcpy(wrk1, Cp, sizeSub * n_remainder * sizeof(CTYPE));
    memset(Cp, 0, sizeSub * nx * sizeof(CTYPE));
    FN(gemm_nn)(sizeSub, nx, nx, (CTYPE)1, wrk1, wrk2, (CTYPE)0, Cp);

cleanup:
    safe_free((void**)&eigvals_all);
    safe_free((void**)&sig_all);
    safe_free((void**)&indices);
    safe_free((void**)&tau);
    safe_free((void**)&VR);
#ifdef CTYPE_IS_COMPLEX
    safe_free((void**)&alpha);
    safe_free((void**)&beta);
#else
    safe_free((void**)&alphar);
    safe_free((void**)&alphai);
    safe_free((void**)&beta);
#endif
}

#undef CONCAT2
#undef CONCAT
#undef FN
#undef MIN
#undef CABS
#undef CREAL
