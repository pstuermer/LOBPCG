/**
 * @file ortho_indefinite_impl.inc
 * @brief Type-generic indefinite B-orthogonalization for ilobpcg
 *
 * B-orthogonalizes U against V when B is indefinite (can have ±1 signature).
 * Uses signature-weighted projection: U = U - V*sig*(V^H*B*U)
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
  #define CREAL(x) creal(x)
#else
  #define CABS(x) fabs(x)
  #define CREAL(x) (x)
#endif

/* Fill lower triangle from upper for symmetric/Hermitian matrix */
static void FN(fill_lower_oi)(uint64_t n, CTYPE *A) {
    for (uint64_t j = 0; j < n; j++)
        for (uint64_t i = j + 1; i < n; i++)
#ifdef CTYPE_IS_COMPLEX
            A[i + j*n] = conj(A[j + i*n]);
#else
            A[i + j*n] = A[j + i*n];
#endif
}

/* Compute Frobenius norm of m x n matrix */
static RTYPE FN(matrix_nrm)(uint64_t m, uint64_t n, const CTYPE *A) {
    RTYPE sum = 0;
    for (uint64_t i = 0; i < m * n; i++) {
        RTYPE val = CABS(A[i]);
        sum += val * val;
    }
    return sqrt(sum);
}

/* Compute ||G - I||_F from upper triangle of Hermitian G (n x n, lda = n) */
static RTYPE FN(ortho_err_upper)(uint64_t n, const CTYPE *G) {
    RTYPE sum = 0;
    for (uint64_t j = 0; j < n; j++) {
        /* Diagonal: |G_jj - 1|^2 */
        RTYPE d = CABS(G[j + j * n] - (CTYPE)1);
        sum += d * d;
        /* Off-diagonal upper triangle: 2*|G_ij|^2 (accounts for lower) */
        for (uint64_t i = 0; i < j; i++) {
            RTYPE v = CABS(G[i + j * n]);
            sum += 2 * v * v;
        }
    }
    return sqrt(sum);
}

/**
 * ortho_indefinite - B-orthogonalize U against V with indefinite B
 *
 * @param m         Number of rows (problem size)
 * @param n_u       Number of columns of U (candidate basis)
 * @param n_v       Number of columns of V (external basis, already B-ortho)
 * @param eps_ortho Tolerance for orthogonality check
 * @param eps_drop  Drop tolerance for svqb
 * @param U         [in/out] m x n_u candidate basis, B-orthonormalized on output
 * @param V         [in] m x n_v B-orthonormal external basis
 * @param sig       [in] n_v x n_v signature matrix. If NULL, computed as V^H*B*V
 * @param wrk1      Workspace: m * max(n_u, n_v) for B*V or B*U storage
 * @param wrk2      Workspace: m * max(n_u, n_v) for intermediate storage
 * @param wrk3      Workspace: max(n_v * n_u, n_v * n_v, n_u * n_u) for coef matrices
 * @param B         Indefinite B operator (NULL = identity)
 *
 * @return Number of columns dropped from U (always 0 for now)
 *
 * Algorithm:
 *   1. Compute sig = V^H*B*V if not provided
 *   2. For i = 0..2 (outer iterations):
 *      a. coef = V^H * B * U
 *      b. tmp = sig * coef
 *      c. U = U - V * tmp
 *      d. B-orthonormalize U via svqb (inner iterations)
 *      e. Check ||V^H*B*U||_F convergence
 */
uint64_t FN(ortho_indefinite)(
    const uint64_t m,
    const uint64_t n_u,
    const uint64_t n_v,
    const RTYPE eps_ortho,
    const RTYPE eps_drop,
    CTYPE *restrict U,
    CTYPE *restrict V,
    CTYPE *restrict sig,
    CTYPE *restrict wrk1,
    CTYPE *restrict wrk2,
    CTYPE *restrict wrk3,
    LINOP *B)
{
    if (0 == n_u || 0 == n_v) return 0;
    if (m < n_u + n_v) return 0;

    const uint64_t max_outer = 3;
    const uint64_t max_inner = 3;

    /* Allocate local signature matrix if not provided */
    CTYPE *sig_local = NULL;
    if (NULL == sig) {
        sig_local = xcalloc(n_v * n_v, sizeof(CTYPE));
        sig = sig_local;

        /* Compute sig = V^H * B * V */
        FN(gram_self)(V, m, n_v, B, sig, n_v, wrk1);
        FN(fill_lower_oi)(n_v, sig);
    }

    /* Compute ||B*V||_F for relative error calculation */
    RTYPE BV_norm;
    if (NULL == B) {
        BV_norm = FN(matrix_nrm)(m, n_v, V);
    } else {
        if (NULL == sig_local) {
            /* sig was provided externally, need to compute B*V */
            FN(apply_block_op)(B, V, wrk1, m, n_v);
        }
        /* B*V already in wrk1 from gram_self above (if sig_local != NULL) */
        BV_norm = FN(matrix_nrm)(m, n_v, wrk1);
    }
    if (BV_norm < eps_ortho) BV_norm = (RTYPE)1;

    /* Outer iterations: project out V component and re-orthonormalize */
    for (uint64_t outer = 0; outer < max_outer; outer++) {

        /* coef = V^H * B * U (n_v x n_u matrix in wrk3) */
        FN(gram_cross)(V, n_v, U, n_u, m, B, wrk3, n_v, wrk2);

        /* tmp = sig * coef (n_v x n_u matrix in wrk1, reusing space) */
        /* Note: wrk1 has size m*max(n_u,n_v), and n_v*n_u <= m*max(n_u,n_v) typically */
        CTYPE *tmp = wrk1;  /* Reuse wrk1 for this small matrix */
        FN(gemm_nn)(n_v, n_u, n_v, (CTYPE)1, sig, wrk3, (CTYPE)0, tmp);

        /* U = U - V * tmp */
        /* wrk2 = V * tmp (m x n_u) */
        FN(gemm_nn)(m, n_u, n_v, (CTYPE)1, V, tmp, (CTYPE)0, wrk2);

        /* U = U - wrk2 */
        for (uint64_t i = 0; i < m * n_u; i++)
            U[i] -= wrk2[i];

        /* Inner iterations: B-orthonormalize U via SVQB */
        for (uint64_t inner = 0; inner < max_inner; inner++) {
            /* Call SVQB to B-orthonormalize U */
            /* svqb signature: (m, n, tau, drop, U, wrk1, wrk2, wrk3, B) */
            FN(svqb)(m, n_u, eps_drop, 'n', U, wrk1, wrk2, wrk3, B);

            /* Check orthonormality: ||U^H*B*U - I||_F (upper triangle only) */
            FN(gram_self)(U, m, n_u, B, wrk1, n_u, wrk2);
            RTYPE ortho_err = FN(ortho_err_upper)(n_u, wrk1);

            if (ortho_err < eps_ortho) break;
        }

        /* Check ||V^H*B*U||_F for convergence */
        FN(gram_cross)(V, n_v, U, n_u, m, B, wrk3, n_v, wrk2);

        RTYPE U_norm = FN(matrix_nrm)(m, n_u, U);
        if (U_norm < eps_ortho) U_norm = (RTYPE)1;

        RTYPE VU_err = FN(matrix_nrm)(n_v, n_u, wrk3) / (BV_norm * U_norm);
        if (VU_err < eps_ortho) break;
    }

    if (NULL != sig_local) {
        safe_free((void**)&sig_local);
    }

    return 0;  /* No column dropping implemented yet */
}

#undef CONCAT2
#undef CONCAT
#undef FN
#undef MAX
#undef CABS
#undef CREAL
