/**
 * @file svqb_mat_impl.inc
 * @brief Type-generic SVQB orthogonalization with explicit dense matrix
 *
 * Like svqb, but takes a dense m x m matrix instead of a LinearOperator.
 * Used in ilobpcg where the indefinite metric is an explicit matrix.
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

/* Fill lower triangle from upper for symmetric/Hermitian matrix */
static void FN(fill_lower_sm)(uint64_t n, CTYPE *A) {
    for (uint64_t j = 0; j < n; j++)
        for (uint64_t i = j + 1; i < n; i++)
#ifdef CTYPE_IS_COMPLEX
            A[i + j*n] = conj(A[j + i*n]);
#else
            A[i + j*n] = A[j + i*n];
#endif
}

/**
 * svqb_mat - SVQB orthogonalization with explicit dense matrix
 *
 * @param m     Number of rows (problem size)
 * @param n     Number of columns of U
 * @param tau   Drop tolerance for eigenvalues
 * @param drop  Drop mode ('n' = no drop, 'y' = drop small columns)
 * @param U     [in/out] m x n matrix, orthonormalized on output w.r.t. mat
 * @param mat   [in] m x m dense matrix (the metric)
 * @param wrk1  Workspace >= max(n*n, m*n)
 * @param wrk2  Workspace >= max(n*n, m*n)
 * @param wrk3  Workspace >= max(n*n, m*n)
 *
 * @return Number of columns retained
 *
 * Algorithm:
 *   1. Gram: G = U^H * mat * U  (two GEMMs: tmp = mat*U, G = U^H*tmp)
 *   2. Diagonal scaling: D_ii = 1/sqrt(|G_ii|)
 *   3. Normalized Gram: G̃ = D * G * D
 *   4. Eigensolve G̃ = V Λ V^H
 *   5. Threshold: τ * max(|λ_i|)
 *   6. Reconstruct: U = U * D * V * diag(1/sqrt(max(|λ_i|, thresh)))
 */
uint64_t FN(svqb_mat)(const uint64_t m, const uint64_t n, const RTYPE tau,
                      const char drop, CTYPE *restrict U, CTYPE *restrict mat,
                      CTYPE *restrict wrk1, CTYPE *restrict wrk2,
                      CTYPE *restrict wrk3) {
    if (0 == n) return 0;
    (void)drop;  /* drop='y' not implemented */

    RTYPE *eigvals = xcalloc(n, sizeof(RTYPE));
    RTYPE *D = xcalloc(n, sizeof(RTYPE));

    /* 1. Gram matrix G = U^H * mat * U via two GEMMs */
    /* wrk2 = mat * U  (m x n) */
    FN(gemm_nn)(m, n, m, (CTYPE)1, mat, U, (CTYPE)0, wrk2);
    /* wrk1 = U^H * wrk2  (n x n) */
#ifdef CTYPE_IS_COMPLEX
    FN(gemm_hn)(n, n, m, (CTYPE)1, U, wrk2, (CTYPE)0, wrk1);
#else
    FN(gemm_tn)(n, n, m, (CTYPE)1, U, wrk2, (CTYPE)0, wrk1);
#endif
    FN(fill_lower_sm)(n, wrk1);

    /* 2. Diagonal scaling D_ii = 1/sqrt(|G_ii|) */
    for (uint64_t i = 0; i < n; i++) {
        RTYPE gii = CABS(wrk1[i + i*n]);
        D[i] = (gii > 0) ? (RTYPE)1/sqrt(gii) : (RTYPE)1;
    }

    /* 3. Scale: G_scaled = D*G*D */
    for (uint64_t j = 0; j < n; j++)
        for (uint64_t i = 0; i < n; i++)
            wrk1[i + j*n] *= D[i] * D[j];

    /* 4. Eigensolve */
    if (FN(eig)(n, wrk1, eigvals) != 0) {
        safe_free((void**)&eigvals); safe_free((void**)&D); return n;
    }

    /* 5. Threshold */
    RTYPE maxeig = 0;
    for (uint64_t i = 0; i < n; i++)
        if (fabs(eigvals[i]) > maxeig) maxeig = fabs(eigvals[i]);
    RTYPE thresh = tau * maxeig;

    /* 6. Transform: U_new = U * D * V * D_final */
    /*    D_final_ii = 1/sqrt(max(|λ_i|, thresh)) */
    /* T = D*V in wrk2 */
    for (uint64_t j = 0; j < n; j++)
        for (uint64_t i = 0; i < n; i++)
            wrk2[i + j*n] = D[i] * wrk1[i + j*n];

    /* T2 = T * D_final in wrk3 */
    for (uint64_t j = 0; j < n; j++) {
        RTYPE df = (RTYPE)1/sqrt(MAX(fabs(eigvals[j]), thresh));
        for (uint64_t i = 0; i < n; i++)
            wrk3[i + j*n] = wrk2[i + j*n] * df;
    }

    /* U = U * T2 */
    FN(gemm_nn)(m, n, n, (CTYPE)1, U, wrk3, (CTYPE)0, wrk2);
    memcpy(U, wrk2, m*n*sizeof(CTYPE));

    safe_free((void**)&eigvals); safe_free((void**)&D);
    return n;
}

#undef CONCAT2
#undef CONCAT
#undef FN
#undef MAX
#undef CABS
