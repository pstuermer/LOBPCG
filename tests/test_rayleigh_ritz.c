/**
 * @file test_rayleigh_ritz.c
 * @brief Tests for standard and modified Rayleigh-Ritz procedures
 *
 * Uses reference test data from LREP_post (4x4 and 6x6 dense symmetric matrices).
 * Verification checks Rayleigh quotient diagonality and B-orthonormality.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include "types.h"
#include "lobpcg.h"
#include "lobpcg/linop.h"
#include "lobpcg/blas_wrapper.h"
#include "lobpcg/memory.h"

#include "test_macros.h"

/* ================================================================
 * Dense matvec helpers
 * ================================================================ */

typedef struct { uint64_t n; f64 *A; } dense_ctx_d_t;
typedef struct { uint64_t n; c64 *A; } dense_ctx_z_t;

static void dense_matvec_d(const LinearOperator_d_t *op,
                           f64 *restrict x, f64 *restrict y) {
    dense_ctx_d_t *ctx = (dense_ctx_d_t *)op->ctx;
    d_gemm_nn(ctx->n, 1, ctx->n, 1.0, ctx->A, x, 0.0, y);
}

static void dense_matvec_z(const LinearOperator_z_t *op,
                           c64 *restrict x, c64 *restrict y) {
    dense_ctx_z_t *ctx = (dense_ctx_z_t *)op->ctx;
    z_gemm_nn(ctx->n, 1, ctx->n, (c64)1, ctx->A, x, (c64)0, y);
}

/* Scale matvec for B = scale*I */
typedef struct { uint64_t n; f64 scale; } scale_ctx_d_t;

static void scale_matvec_d(const LinearOperator_d_t *op,
                           f64 *restrict x, f64 *restrict y) {
    scale_ctx_d_t *ctx = (scale_ctx_d_t *)op->ctx;
    for (uint64_t i = 0; i < ctx->n; i++) y[i] = ctx->scale * x[i];
}

/* ================================================================
 * Reference data: 4x4 symmetric matrix
 * ================================================================ */
static const f64 A4x4[16] = {
    4.0, 1.0, 2.0, 0.0,
    1.0, 3.0, 0.0, 1.0,
    2.0, 0.0, 5.0, 2.0,
    0.0, 1.0, 2.0, 6.0
};

/* S for standard RR: 4x2, column-major */
static const f64 S4x2_d[8] = {
    1.0, -1.0, 1.0, -1.0,   /* col 0 */
    1.0, 1.0, -1.0, -2.0    /* col 1 */
};

/* Expected eigenvalues from reference 4x4 standard RR */
static const f64 eigVal_ref_4x4[2] = { 4.270248, 5.507529 };

/* Expected X_new = S * Cx: 4x2, column-major (up to column sign) */
static const f64 Xnew_ref_4x4[8] = {
    0.326799989, -0.658521176, 0.658521176, -0.160939395,  /* col 0 */
    0.475957037, 0.218703777, -0.218703777, -0.823287444   /* col 1 */
};

/* ================================================================
 * Reference data: 6x6 symmetric matrix
 * ================================================================ */
static const f64 A6x6[36] = {
    4.0, 1.0, 2.0, 0.0, 1.0, 0.5,
    1.0, 3.0, 0.0, 1.0, 0.5, 0.0,
    2.0, 0.0, 5.0, 2.0, 1.0, 1.0,
    0.0, 1.0, 2.0, 6.0, 1.5, 0.0,
    1.0, 0.5, 1.0, 1.5, 5.0, 2.0,
    0.5, 0.0, 1.0, 0.0, 2.0, 4.0
};

/* S for modified RR: 6x4, column-major */
static const f64 S6x4_d[24] = {
    1.0, -1.0, 1.0, -1.0, 1.0, -1.0,   /* col 0 */
    1.0, 1.0, -1.0, -2.0, 1.0, 2.0,    /* col 1 */
    1.0, 1.0, 1.0, -1.0, -1.0, -1.0,   /* col 2 */
    2.0, 1.0, -2.0, -1.0, -1.0, 2.0    /* col 3 */
};

/* ================================================================
 * Helper: column-sign-adjusted error between two real matrices
 * For each column j, checks both Cx[:,j] ≈ ref[:,j] and Cx[:,j] ≈ -ref[:,j]
 * ================================================================ */
static f64 cx_error_signed(uint64_t m, uint64_t n, const f64 *Cx, const f64 *ref) {
    f64 max_err = 0;
    for (uint64_t j = 0; j < n; j++) {
        f64 err_pos = 0, err_neg = 0;
        for (uint64_t i = 0; i < m; i++) {
            f64 ep = fabs(Cx[i + j*m] - ref[i + j*m]);
            f64 en = fabs(Cx[i + j*m] + ref[i + j*m]);
            if (ep > err_pos) err_pos = ep;
            if (en > err_neg) err_neg = en;
        }
        f64 col_err = (err_pos < err_neg) ? err_pos : err_neg;
        if (col_err > max_err) max_err = col_err;
    }
    return max_err;
}

/* ================================================================
 * Helper: check Rayleigh quotient X^T*A*X ≈ diag(eigVal)
 * Returns ||X^T*A*X - diag(eigVal)||_F
 * ================================================================ */
static f64 rayleigh_diag_d(uint64_t n, uint64_t nev, const f64 *A_mat,
                           const f64 *X, const f64 *eigVal) {
    f64 *AX = xcalloc(n * nev, sizeof(f64));
    f64 *G  = xcalloc(nev * nev, sizeof(f64));
    d_gemm_nn(n, nev, n, 1.0, A_mat, X, 0.0, AX);
    d_gemm_tn(nev, nev, n, 1.0, X, AX, 0.0, G);

    /* Subtract diag(eigVal) */
    for (uint64_t i = 0; i < nev; i++) G[i + i*nev] -= eigVal[i];

    f64 err = d_nrm2(nev * nev, G);
    safe_free((void**)&AX); safe_free((void**)&G);
    return err;
}

static f64 rayleigh_diag_z(uint64_t n, uint64_t nev, const c64 *A_mat,
                           const c64 *X, const f64 *eigVal) {
    c64 *AX = xcalloc(n * nev, sizeof(c64));
    c64 *G  = xcalloc(nev * nev, sizeof(c64));
    z_gemm_nn(n, nev, n, (c64)1, A_mat, X, (c64)0, AX);
    z_gemm_hn(nev, nev, n, (c64)1, X, AX, (c64)0, G);

    for (uint64_t i = 0; i < nev; i++) G[i + i*nev] -= eigVal[i];

    f64 err = z_nrm2(nev * nev, G);
    safe_free((void**)&AX); safe_free((void**)&G);
    return err;
}

/* ================================================================
 * Helper: B-orthonormality ||X^T*B*X - I||_F  (B=NULL → ||X^T*X - I||_F)
 * ================================================================ */
static f64 ortho_self_d(uint64_t n, uint64_t nev, const f64 *X) {
    f64 *G = xcalloc(nev * nev, sizeof(f64));
    d_syrk(n, nev, 1.0, X, 0.0, G);
    for (uint64_t j = 0; j < nev; j++)
        for (uint64_t i = j; i < nev; i++) {
            if (i == j) G[i + j*nev] -= 1.0;
            else G[i + j*nev] = G[j + i*nev];
        }
    f64 err = d_nrm2(nev * nev, G);
    safe_free((void**)&G);
    return err;
}

static f64 ortho_self_z(uint64_t n, uint64_t nev, const c64 *X) {
    c64 *G = xcalloc(nev * nev, sizeof(c64));
    z_herk(n, nev, 1.0, X, 0.0, G);
    for (uint64_t j = 0; j < nev; j++)
        for (uint64_t i = j; i < nev; i++) {
            if (i == j) G[i + j*nev] -= 1.0;
            else G[i + j*nev] = conj(G[j + i*nev]);
        }
    f64 err = z_nrm2(nev * nev, G);
    safe_free((void**)&G);
    return err;
}

/* ================================================================
 * Helper: cross-orthogonality ||X^T * P||_F
 * ================================================================ */
static f64 ortho_cross_d(uint64_t n, uint64_t nx, const f64 *X, const f64 *P) {
    f64 *G = xcalloc(nx * nx, sizeof(f64));
    d_gemm_tn(nx, nx, n, 1.0, X, P, 0.0, G);
    f64 err = d_nrm2(nx * nx, G);
    safe_free((void**)&G);
    return err;
}

static f64 ortho_cross_z(uint64_t n, uint64_t nx, const c64 *X, const c64 *P) {
    c64 *G = xcalloc(nx * nx, sizeof(c64));
    z_gemm_hn(nx, nx, n, (c64)1, X, P, (c64)0, G);
    f64 err = z_nrm2(nx * nx, G);
    safe_free((void**)&G);
    return err;
}

/* ================================================================
 * Test 1: Standard RR, double, 4x4 dense
 * ================================================================ */
TEST(d_rr_4x4) {
    const uint64_t n = 4, nev = 2;

    dense_ctx_d_t *ctx = xcalloc(1, sizeof(dense_ctx_d_t));
    ctx->n = n;
    ctx->A = xcalloc(n * n, sizeof(f64));
    memcpy(ctx->A, A4x4, n * n * sizeof(f64));

    LinearOperator_d_t A = { .rows = n, .cols = n, .matvec = dense_matvec_d,
                             .ctx = (linop_ctx_t *)ctx };

    f64 *S    = xcalloc(n * nev, sizeof(f64));
    f64 *Cx   = xcalloc(nev * nev, sizeof(f64));
    f64 *eigVal = xcalloc(nev, sizeof(f64));
    f64 *wrk1 = xcalloc(nev * nev, sizeof(f64));
    f64 *wrk2 = xcalloc(n * nev, sizeof(f64));
    f64 *wrk3 = xcalloc(nev * nev, sizeof(f64));

    memcpy(S, S4x2_d, n * nev * sizeof(f64));

    d_rayleigh_ritz(n, nev, S, Cx, eigVal, wrk1, wrk2, wrk3, &A, NULL);

    /* Check eigenvalues */
    ASSERT_NEAR(eigVal[0], eigVal_ref_4x4[0], 1e-4);
    ASSERT_NEAR(eigVal[1], eigVal_ref_4x4[1], 1e-4);

    /* Check X_new = S * Cx matches reference (up to column sign) */
    f64 *X_new = xcalloc(n * nev, sizeof(f64));
    d_gemm_nn(n, nev, nev, 1.0, S, Cx, 0.0, X_new);

    f64 xn_err = cx_error_signed(n, nev, X_new, Xnew_ref_4x4);
    printf("xn=%.2e ", xn_err);
    ASSERT(xn_err < 1e-6);

    /* Check X_new^T*A*X_new = diag(eigVal) */
    f64 rq_err = rayleigh_diag_d(n, nev, ctx->A, X_new, eigVal);
    ASSERT(rq_err < 1e-10);

    /* Check orthonormality */
    f64 orth_err = ortho_self_d(n, nev, X_new);
    ASSERT(orth_err < 1e-10);

    safe_free((void**)&S);    safe_free((void**)&Cx);   safe_free((void**)&eigVal);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3);
    safe_free((void**)&X_new);
    safe_free((void**)&ctx->A); safe_free((void**)&ctx);
}

/* ================================================================
 * Test 2: Standard RR, complex double, 4x4 — complex S
 * ================================================================ */
TEST(z_rr_4x4) {
    const uint64_t n = 4, nev = 2;

    dense_ctx_z_t *ctx = xcalloc(1, sizeof(dense_ctx_z_t));
    ctx->n = n;
    ctx->A = xcalloc(n * n, sizeof(c64));
    for (uint64_t i = 0; i < n * n; i++) ctx->A[i] = A4x4[i] + 0*I;

    LinearOperator_z_t A = { .rows = n, .cols = n, .matvec = dense_matvec_z,
                             .ctx = (linop_ctx_t *)ctx };

    c64 *S    = xcalloc(n * nev, sizeof(c64));
    c64 *Cx   = xcalloc(nev * nev, sizeof(c64));
    f64 *eigVal = xcalloc(nev, sizeof(f64));
    c64 *wrk1 = xcalloc(nev * nev, sizeof(c64));
    c64 *wrk2 = xcalloc(n * nev, sizeof(c64));
    c64 *wrk3 = xcalloc(nev * nev, sizeof(c64));

    /* Complex S with nonzero imaginary parts */
    S[0] = 1.0 + 0.5*I;  S[1] = -1.0 + 0.3*I;
    S[2] = 1.0 - 0.2*I;  S[3] = -1.0 + 0.1*I;
    S[4] = 1.0 - 0.3*I;  S[5] = 1.0 + 0.5*I;
    S[6] = -1.0 + 0.4*I; S[7] = -2.0 - 0.2*I;

    z_rayleigh_ritz(n, nev, S, Cx, eigVal, wrk1, wrk2, wrk3, &A, NULL);

    c64 *X_new = xcalloc(n * nev, sizeof(c64));
    z_gemm_nn(n, nev, nev, (c64)1, S, Cx, (c64)0, X_new);

    /* Check X^H*A*X = diag(eigVal) */
    f64 rq_err = rayleigh_diag_z(n, nev, ctx->A, X_new, eigVal);
    printf("rq=%.2e ", rq_err);
    ASSERT(rq_err < 1e-10);

    /* Check orthonormality */
    f64 orth_err = ortho_self_z(n, nev, X_new);
    ASSERT(orth_err < 1e-10);

    /* Eigenvalues should be positive and sorted */
    ASSERT(eigVal[0] > 0);
    ASSERT(eigVal[0] < eigVal[1]);

    safe_free((void**)&S);    safe_free((void**)&Cx);   safe_free((void**)&eigVal);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3);
    safe_free((void**)&X_new);
    safe_free((void**)&ctx->A); safe_free((void**)&ctx);
}

/* ================================================================
 * Test 3: Standard RR with B=2I (double)
 * ================================================================ */
TEST(d_rr_4x4_with_B) {
    const uint64_t n = 4, nev = 2;

    dense_ctx_d_t *ctxA = xcalloc(1, sizeof(dense_ctx_d_t));
    ctxA->n = n;
    ctxA->A = xcalloc(n * n, sizeof(f64));
    memcpy(ctxA->A, A4x4, n * n * sizeof(f64));

    scale_ctx_d_t *ctxB = xcalloc(1, sizeof(scale_ctx_d_t));
    ctxB->n = n;
    ctxB->scale = 2.0;

    LinearOperator_d_t A = { .rows = n, .cols = n, .matvec = dense_matvec_d,
                             .ctx = (linop_ctx_t *)ctxA };
    LinearOperator_d_t B = { .rows = n, .cols = n, .matvec = scale_matvec_d,
                             .ctx = (linop_ctx_t *)ctxB };

    f64 *S    = xcalloc(n * nev, sizeof(f64));
    f64 *Cx   = xcalloc(nev * nev, sizeof(f64));
    f64 *eigVal = xcalloc(nev, sizeof(f64));
    f64 *wrk1 = xcalloc(nev * nev, sizeof(f64));
    f64 *wrk2 = xcalloc(n * nev, sizeof(f64));
    f64 *wrk3 = xcalloc(nev * nev, sizeof(f64));

    memcpy(S, S4x2_d, n * nev * sizeof(f64));

    d_rayleigh_ritz(n, nev, S, Cx, eigVal, wrk1, wrk2, wrk3, &A, &B);

    /* Generalized eigenvalues should be half of standard */
    ASSERT_NEAR(eigVal[0], eigVal_ref_4x4[0] / 2.0, 1e-4);
    ASSERT_NEAR(eigVal[1], eigVal_ref_4x4[1] / 2.0, 1e-4);

    /* Check X_new^T*A*X_new = diag(eigVal) */
    f64 *X_new = xcalloc(n * nev, sizeof(f64));
    d_gemm_nn(n, nev, nev, 1.0, S, Cx, 0.0, X_new);

    f64 rq_err = rayleigh_diag_d(n, nev, ctxA->A, X_new, eigVal);
    printf("rq=%.2e ", rq_err);
    /* For generalized: X^T*A*X = Lambda * X^T*B*X = Lambda * 2*I
     * So X^T*A*X should be diag(2*eigVal) */
    /* Actually, let me compute it directly */

    /* For gen eig: X^T*A*X = diag(eigVal) * X^T*B*X
     * If X is B-orthonormal: X^T*B*X = I, then X^T*A*X = diag(eigVal)
     * But eigVal here are gen eigenvalues, so X^T*A*X = diag(eigVal) only if
     * X is B-orthonormal. RR produces B-orthonormal X, so this should hold. */

    /* Check B-orthonormality: X^T * (2I) * X = I  → X^T*X = 0.5*I */
    f64 *G = xcalloc(nev * nev, sizeof(f64));
    d_syrk(n, nev, 1.0, X_new, 0.0, G);
    for (uint64_t j = 0; j < nev; j++)
        for (uint64_t i = j; i < nev; i++) {
            if (i == j) G[i + j*nev] -= 0.5;
            else G[i + j*nev] = G[j + i*nev];
        }
    f64 borth = d_nrm2(nev * nev, G);
    printf("borth=%.2e ", borth);
    ASSERT(borth < 1e-10);

    safe_free((void**)&S);    safe_free((void**)&Cx);   safe_free((void**)&eigVal);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3);
    safe_free((void**)&X_new); safe_free((void**)&G);
    safe_free((void**)&ctxA->A); safe_free((void**)&ctxA); safe_free((void**)&ctxB);
}

/* ================================================================
 * Test 4: Modified RR, useOrtho=1, double, 6x6 dense
 * ================================================================ */
TEST(d_rr_modified_ortho) {
    const uint64_t n = 6, nx = 2, mult = 2;
    const uint64_t sizeSub = mult * nx;
    const uint64_t n_rem = (mult - 1) * nx;

    dense_ctx_d_t *ctx = xcalloc(1, sizeof(dense_ctx_d_t));
    ctx->n = n;
    ctx->A = xcalloc(n * n, sizeof(f64));
    memcpy(ctx->A, A6x6, n * n * sizeof(f64));

    LinearOperator_d_t A = { .rows = n, .cols = n, .matvec = dense_matvec_d,
                             .ctx = (linop_ctx_t *)ctx };

    f64 *S    = xcalloc(n * sizeSub, sizeof(f64));
    f64 *Cx   = xcalloc(sizeSub * sizeSub, sizeof(f64));
    f64 *Cp   = xcalloc(sizeSub * n_rem, sizeof(f64));
    f64 *eigVal = xcalloc(nx, sizeof(f64));
    f64 *wrk1 = xcalloc(sizeSub * sizeSub, sizeof(f64));
    f64 *wrk2 = xcalloc(n * sizeSub, sizeof(f64));
    f64 *wrk3 = xcalloc(sizeSub * sizeSub, sizeof(f64));

    memcpy(S, S6x4_d, n * sizeSub * sizeof(f64));

    uint8_t useOrtho = 1;
    d_rayleigh_ritz_modified(n, nx, mult, 0, 0, &useOrtho,
                             S, NULL, wrk1, wrk2, wrk3, Cx, Cp, eigVal, &A, NULL);

    ASSERT(1 == useOrtho);

    /* Check X_new^T*A*X_new = diag(eigVal) */
    f64 *X_new = xcalloc(n * nx, sizeof(f64));
    d_gemm_nn(n, nx, sizeSub, 1.0, S, Cx, 0.0, X_new);
    f64 rq_err = rayleigh_diag_d(n, nx, ctx->A, X_new, eigVal);
    printf("rq=%.2e ", rq_err);
    ASSERT(rq_err < 1e-8);

    /* Eigenvalues positive and sorted */
    ASSERT(eigVal[0] > 0);
    ASSERT(eigVal[0] < eigVal[1]);

    safe_free((void**)&S);    safe_free((void**)&Cx);   safe_free((void**)&Cp);
    safe_free((void**)&eigVal);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3);
    safe_free((void**)&X_new);
    safe_free((void**)&ctx->A); safe_free((void**)&ctx);
}

/* ================================================================
 * Test 5: Modified RR, useOrtho=1, complex double, 6x6 dense
 * ================================================================ */
TEST(z_rr_modified_ortho) {
    const uint64_t n = 6, nx = 2, mult = 2;
    const uint64_t sizeSub = mult * nx;
    const uint64_t n_rem = (mult - 1) * nx;

    dense_ctx_z_t *ctx = xcalloc(1, sizeof(dense_ctx_z_t));
    ctx->n = n;
    ctx->A = xcalloc(n * n, sizeof(c64));
    for (uint64_t i = 0; i < n * n; i++) ctx->A[i] = A6x6[i] + 0*I;

    LinearOperator_z_t A = { .rows = n, .cols = n, .matvec = dense_matvec_z,
                             .ctx = (linop_ctx_t *)ctx };

    c64 *S    = xcalloc(n * sizeSub, sizeof(c64));
    c64 *Cx   = xcalloc(sizeSub * sizeSub, sizeof(c64));
    c64 *Cp   = xcalloc(sizeSub * n_rem, sizeof(c64));
    f64 *eigVal = xcalloc(nx, sizeof(f64));
    c64 *wrk1 = xcalloc(sizeSub * sizeSub, sizeof(c64));
    c64 *wrk2 = xcalloc(n * sizeSub, sizeof(c64));
    c64 *wrk3 = xcalloc(sizeSub * sizeSub, sizeof(c64));

    /* Complex S with nonzero imaginary parts */
    for (uint64_t i = 0; i < n * sizeSub; i++)
        S[i] = S6x4_d[i] + 0.3 * ((f64)(i % 7) - 3.0) * I;

    uint8_t useOrtho = 1;
    z_rayleigh_ritz_modified(n, nx, mult, 0, 0, &useOrtho,
                             S, NULL, wrk1, wrk2, wrk3, Cx, Cp, eigVal, &A, NULL);

    ASSERT(1 == useOrtho);

    /* Check X^H*A*X = diag(eigVal) */
    c64 *X_new = xcalloc(n * nx, sizeof(c64));
    z_gemm_nn(n, nx, sizeSub, (c64)1, S, Cx, (c64)0, X_new);
    f64 rq_err = rayleigh_diag_z(n, nx, ctx->A, X_new, eigVal);
    printf("rq=%.2e ", rq_err);
    ASSERT(rq_err < 1e-8);

    /* Eigenvalues positive and sorted */
    ASSERT(eigVal[0] > 0);
    ASSERT(eigVal[0] < eigVal[1]);

    safe_free((void**)&S);    safe_free((void**)&Cx);   safe_free((void**)&Cp);
    safe_free((void**)&eigVal);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3);
    safe_free((void**)&X_new);
    safe_free((void**)&ctx->A); safe_free((void**)&ctx);
}

/* ================================================================
 * Test 6: Modified RR, useOrtho=0 (Cholesky), double, 6x6 dense
 * ================================================================ */
TEST(d_rr_modified_chol) {
    const uint64_t n = 6, nx = 2, mult = 2;
    const uint64_t sizeSub = mult * nx;
    const uint64_t n_rem = (mult - 1) * nx;

    dense_ctx_d_t *ctx = xcalloc(1, sizeof(dense_ctx_d_t));
    ctx->n = n;
    ctx->A = xcalloc(n * n, sizeof(f64));
    memcpy(ctx->A, A6x6, n * n * sizeof(f64));

    LinearOperator_d_t A = { .rows = n, .cols = n, .matvec = dense_matvec_d,
                             .ctx = (linop_ctx_t *)ctx };

    f64 *S    = xcalloc(n * sizeSub, sizeof(f64));
    f64 *Cx   = xcalloc(sizeSub * sizeSub, sizeof(f64));
    f64 *Cp   = xcalloc(sizeSub * n_rem, sizeof(f64));
    f64 *eigVal = xcalloc(nx, sizeof(f64));
    f64 *wrk1 = xcalloc(sizeSub * sizeSub, sizeof(f64));
    f64 *wrk2 = xcalloc(n * sizeSub, sizeof(f64));
    f64 *wrk3 = xcalloc(sizeSub * sizeSub, sizeof(f64));

    memcpy(S, S6x4_d, n * sizeSub * sizeof(f64));

    uint8_t useOrtho = 0;
    d_rayleigh_ritz_modified(n, nx, mult, 0, 0, &useOrtho,
                             S, NULL, wrk1, wrk2, wrk3, Cx, Cp, eigVal, &A, NULL);

    ASSERT(0 == useOrtho);

    /* Check X_new^T*A*X_new = diag(eigVal) */
    f64 *X_new = xcalloc(n * nx, sizeof(f64));
    d_gemm_nn(n, nx, sizeSub, 1.0, S, Cx, 0.0, X_new);
    f64 rq_err = rayleigh_diag_d(n, nx, ctx->A, X_new, eigVal);
    printf("rq=%.2e ", rq_err);
    ASSERT(rq_err < 1e-8);

    /* Check P_new orthogonal to X_new */
    f64 *P_new = xcalloc(n * nx, sizeof(f64));
    d_gemm_nn(n, nx, sizeSub, 1.0, S, Cp, 0.0, P_new);
    f64 orth = ortho_cross_d(n, nx, X_new, P_new);
    printf("orth=%.2e ", orth);
    ASSERT(orth < 1e-8);

    /* Eigenvalues positive and sorted */
    ASSERT(eigVal[0] > 0);
    ASSERT(eigVal[0] < eigVal[1]);

    safe_free((void**)&S);    safe_free((void**)&Cx);   safe_free((void**)&Cp);
    safe_free((void**)&eigVal);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3);
    safe_free((void**)&X_new); safe_free((void**)&P_new);
    safe_free((void**)&ctx->A); safe_free((void**)&ctx);
}

/* ================================================================
 * Test 7: Modified RR, useOrtho=0 (Cholesky), complex double
 * ================================================================ */
TEST(z_rr_modified_chol) {
    const uint64_t n = 6, nx = 2, mult = 2;
    const uint64_t sizeSub = mult * nx;
    const uint64_t n_rem = (mult - 1) * nx;

    dense_ctx_z_t *ctx = xcalloc(1, sizeof(dense_ctx_z_t));
    ctx->n = n;
    ctx->A = xcalloc(n * n, sizeof(c64));
    for (uint64_t i = 0; i < n * n; i++) ctx->A[i] = A6x6[i] + 0*I;

    LinearOperator_z_t A = { .rows = n, .cols = n, .matvec = dense_matvec_z,
                             .ctx = (linop_ctx_t *)ctx };

    c64 *S    = xcalloc(n * sizeSub, sizeof(c64));
    c64 *Cx   = xcalloc(sizeSub * sizeSub, sizeof(c64));
    c64 *Cp   = xcalloc(sizeSub * n_rem, sizeof(c64));
    f64 *eigVal = xcalloc(nx, sizeof(f64));
    c64 *wrk1 = xcalloc(sizeSub * sizeSub, sizeof(c64));
    c64 *wrk2 = xcalloc(n * sizeSub, sizeof(c64));
    c64 *wrk3 = xcalloc(sizeSub * sizeSub, sizeof(c64));

    /* Complex S */
    for (uint64_t i = 0; i < n * sizeSub; i++)
        S[i] = S6x4_d[i] + 0.3 * ((f64)(i % 7) - 3.0) * I;

    uint8_t useOrtho = 0;
    z_rayleigh_ritz_modified(n, nx, mult, 0, 0, &useOrtho,
                             S, NULL, wrk1, wrk2, wrk3, Cx, Cp, eigVal, &A, NULL);

    ASSERT(0 == useOrtho);

    /* Check X^H*A*X = diag(eigVal) */
    c64 *X_new = xcalloc(n * nx, sizeof(c64));
    z_gemm_nn(n, nx, sizeSub, (c64)1, S, Cx, (c64)0, X_new);
    f64 rq_err = rayleigh_diag_z(n, nx, ctx->A, X_new, eigVal);
    printf("rq=%.2e ", rq_err);
    ASSERT(rq_err < 1e-8);

    /* Check P_new orthogonal to X_new */
    c64 *P_new = xcalloc(n * nx, sizeof(c64));
    z_gemm_nn(n, nx, sizeSub, (c64)1, S, Cp, (c64)0, P_new);
    f64 orth = ortho_cross_z(n, nx, X_new, P_new);
    printf("orth=%.2e ", orth);
    ASSERT(orth < 1e-8);

    safe_free((void**)&S);    safe_free((void**)&Cx);   safe_free((void**)&Cp);
    safe_free((void**)&eigVal);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3);
    safe_free((void**)&X_new); safe_free((void**)&P_new);
    safe_free((void**)&ctx->A); safe_free((void**)&ctx);
}

/* ================================================================
 * Test 8: Modified RR, mult=3, double, 6x6 dense
 * ================================================================ */
TEST(d_rr_modified_mult3) {
    const uint64_t n = 6, nx = 2, mult = 3;
    const uint64_t sizeSub = mult * nx;  /* 6 */
    const uint64_t n_rem = (mult - 1) * nx;  /* 4 */

    dense_ctx_d_t *ctx = xcalloc(1, sizeof(dense_ctx_d_t));
    ctx->n = n;
    ctx->A = xcalloc(n * n, sizeof(f64));
    memcpy(ctx->A, A6x6, n * n * sizeof(f64));

    LinearOperator_d_t A = { .rows = n, .cols = n, .matvec = dense_matvec_d,
                             .ctx = (linop_ctx_t *)ctx };

    /* S: 6x6, first 4 cols from reference + 2 additional independent vectors */
    f64 *S = xcalloc(n * sizeSub, sizeof(f64));
    memcpy(S, S6x4_d, n * 4 * sizeof(f64));
    /* col 4: {0,1,0,1,0,1} */
    S[1 + 4*n] = 1.0; S[3 + 4*n] = 1.0; S[5 + 4*n] = 1.0;
    /* col 5: {1,0,1,0,1,0} */
    S[0 + 5*n] = 1.0; S[2 + 5*n] = 1.0; S[4 + 5*n] = 1.0;

    f64 *Cx   = xcalloc(sizeSub * sizeSub, sizeof(f64));
    f64 *Cp   = xcalloc(sizeSub * n_rem, sizeof(f64));
    f64 *eigVal = xcalloc(nx, sizeof(f64));
    f64 *wrk1 = xcalloc(sizeSub * sizeSub, sizeof(f64));
    f64 *wrk2 = xcalloc(n * sizeSub, sizeof(f64));
    f64 *wrk3 = xcalloc(sizeSub * sizeSub, sizeof(f64));

    uint8_t useOrtho = 1;
    d_rayleigh_ritz_modified(n, nx, mult, 0, 0, &useOrtho,
                             S, NULL, wrk1, wrk2, wrk3, Cx, Cp, eigVal, &A, NULL);

    ASSERT(1 == useOrtho);

    /* Check X_new^T*A*X_new = diag(eigVal) */
    f64 *X_new = xcalloc(n * nx, sizeof(f64));
    d_gemm_nn(n, nx, sizeSub, 1.0, S, Cx, 0.0, X_new);
    f64 rq_err = rayleigh_diag_d(n, nx, ctx->A, X_new, eigVal);
    printf("rq=%.2e ", rq_err);
    ASSERT(rq_err < 1e-8);

    /* S spans all of R^6, eigenvalues should be the 2 lowest of A */
    ASSERT(eigVal[0] > 0);
    ASSERT(eigVal[0] < eigVal[1]);

    safe_free((void**)&S);    safe_free((void**)&Cx);   safe_free((void**)&Cp);
    safe_free((void**)&eigVal);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3);
    safe_free((void**)&X_new);
    safe_free((void**)&ctx->A); safe_free((void**)&ctx);
}

/* ================================================================ */
int main(void) {
    printf("Standard Rayleigh-Ritz tests:\n");
    RUN(d_rr_4x4);
    RUN(z_rr_4x4);
    RUN(d_rr_4x4_with_B);

    printf("\nModified Rayleigh-Ritz tests:\n");
    RUN(d_rr_modified_ortho);
    RUN(z_rr_modified_ortho);
    RUN(d_rr_modified_chol);
    RUN(z_rr_modified_chol);
    RUN(d_rr_modified_mult3);

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}
