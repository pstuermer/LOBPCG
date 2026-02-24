/**
 * @file test_indefinite_rr.c
 * @brief Tests for indefinite Rayleigh-Ritz procedures (all 4 types)
 *
 * Test categories:
 *   1. Diagonal A + diagonal indefinite B (analytic eigenvalues)
 *   2. Diagonal A + block-permutation B (analytic eigenvalues)
 *   3. Dense A + diagonal indefinite B (structural checks)
 *   4. Dense A + block-permutation B (structural checks)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <complex.h>
#include "types.h"
#include "lobpcg.h"
#include "lobpcg/linop.h"
#include "lobpcg/blas_wrapper.h"
#include "lobpcg/memory.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) static void test_##name(void)
#define RUN(name) do { \
    printf("  %-50s ", #name); \
    test_##name(); \
    printf("[PASS]\n"); \
    tests_passed++; \
} while(0)

#define ASSERT(cond) do { \
    if (!(cond)) { \
        printf("[FAIL] line %d: %s\n", __LINE__, #cond); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define TEST_TOL_D 1e-10
#define TEST_TOL_S 1e-4

/* ================================================================
 * Diagonal matvec helpers
 * ================================================================ */

typedef struct { uint64_t n; f32 *diag; } diag_ctx_s_t;

void diag_matvec_s(const LinearOperator_s_t *op, f32 *restrict x, f32 *restrict y) {
    diag_ctx_s_t *ctx = (diag_ctx_s_t *)op->ctx;
    for (uint64_t i = 0; i < ctx->n; i++) y[i] = ctx->diag[i] * x[i];
}

typedef struct { uint64_t n; f64 *diag; } diag_ctx_d_t;

void diag_matvec_d(const LinearOperator_d_t *op, f64 *restrict x, f64 *restrict y) {
    diag_ctx_d_t *ctx = (diag_ctx_d_t *)op->ctx;
    for (uint64_t i = 0; i < ctx->n; i++) y[i] = ctx->diag[i] * x[i];
}

typedef struct { uint64_t n; f32 *diag; } diag_ctx_c_t;

void diag_matvec_c(const LinearOperator_c_t *op, c32 *restrict x, c32 *restrict y) {
    diag_ctx_c_t *ctx = (diag_ctx_c_t *)op->ctx;
    for (uint64_t i = 0; i < ctx->n; i++) y[i] = ctx->diag[i] * x[i];
}

typedef struct { uint64_t n; f64 *diag; } diag_ctx_z_t;

void diag_matvec_z(const LinearOperator_z_t *op, c64 *restrict x, c64 *restrict y) {
    diag_ctx_z_t *ctx = (diag_ctx_z_t *)op->ctx;
    for (uint64_t i = 0; i < ctx->n; i++) y[i] = ctx->diag[i] * x[i];
}

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

/* ================================================================
 * Block-permutation B operator
 * B = blkdiag({{0,1},{1,0}}, {{0,1},{1,0}}, ...)
 * ================================================================ */

typedef struct { uint64_t m; } perm_blk_ctx_t;

static void perm_blk_matvec_d(const LinearOperator_d_t *op, f64 *x, f64 *y) {
    perm_blk_ctx_t *ctx = (perm_blk_ctx_t *)op->ctx->data;
    uint64_t pairs = ctx->m / 2;
    for (uint64_t i = 0; i < pairs; i++) {
        y[2*i]     = x[2*i + 1];
        y[2*i + 1] = x[2*i];
    }
    if (ctx->m % 2) y[ctx->m - 1] = x[ctx->m - 1];
}

static void perm_blk_matvec_z(const LinearOperator_z_t *op, c64 *x, c64 *y) {
    perm_blk_ctx_t *ctx = (perm_blk_ctx_t *)op->ctx->data;
    uint64_t pairs = ctx->m / 2;
    for (uint64_t i = 0; i < pairs; i++) {
        y[2*i]     = x[2*i + 1];
        y[2*i + 1] = x[2*i];
    }
    if (ctx->m % 2) y[ctx->m - 1] = x[ctx->m - 1];
}

static void perm_blk_cleanup(linop_ctx_t *ctx) {
    if (ctx && ctx->data) safe_free((void**)&ctx->data);
    if (ctx) safe_free((void**)&ctx);
}

static LinearOperator_d_t *create_perm_B_d(uint64_t m) {
    linop_ctx_t *ctx = xcalloc(1, sizeof(linop_ctx_t));
    perm_blk_ctx_t *data = xcalloc(1, sizeof(perm_blk_ctx_t));
    data->m = m;
    ctx->data = data;
    ctx->data_size = sizeof(perm_blk_ctx_t);
    return linop_create_d(m, m, perm_blk_matvec_d, perm_blk_cleanup, ctx);
}

static LinearOperator_z_t *create_perm_B_z(uint64_t m) {
    linop_ctx_t *ctx = xcalloc(1, sizeof(linop_ctx_t));
    perm_blk_ctx_t *data = xcalloc(1, sizeof(perm_blk_ctx_t));
    data->m = m;
    ctx->data = data;
    ctx->data_size = sizeof(perm_blk_ctx_t);
    return linop_create_z(m, m, perm_blk_matvec_z, perm_blk_cleanup, ctx);
}

/* ================================================================
 * Reference data: 6x6 symmetric matrix (column-major)
 * ================================================================ */
static const f64 A6x6[36] = {
    4.0, 1.0, 2.0, 0.0, 1.0, 0.5,
    1.0, 3.0, 0.0, 1.0, 0.5, 0.0,
    2.0, 0.0, 5.0, 2.0, 1.0, 1.0,
    0.0, 1.0, 2.0, 6.0, 1.5, 0.0,
    1.0, 0.5, 1.0, 1.5, 5.0, 2.0,
    0.5, 0.0, 1.0, 0.0, 2.0, 4.0
};

/* ================================================================
 * Helper: B-signature-orthonormality check
 * Returns ||X^H*B*X - diag(sig)||_F
 * ================================================================ */
static f64 B_sig_ortho_d(uint64_t n, uint64_t nev, const f64 *X,
                          LinearOperator_d_t *B, const int8_t *sig, f64 *wrk) {
    /* wrk: needs n*nev for BX */
    for (uint64_t j = 0; j < nev; j++)
        B->matvec(B, (f64*)&X[j * n], &wrk[j * n]);

    f64 *G = xcalloc(nev * nev, sizeof(f64));
    d_gemm_tn(nev, nev, n, 1.0, X, wrk, 0.0, G);

    /* Subtract diag(sig) */
    for (uint64_t i = 0; i < nev; i++) G[i + i * nev] -= (f64)sig[i];

    f64 err = d_nrm2(nev * nev, G);
    safe_free((void**)&G);
    return err;
}

static f64 B_sig_ortho_z(uint64_t n, uint64_t nev, const c64 *X,
                          LinearOperator_z_t *B, const int8_t *sig, c64 *wrk) {
    for (uint64_t j = 0; j < nev; j++)
        B->matvec(B, (c64*)&X[j * n], &wrk[j * n]);

    c64 *G = xcalloc(nev * nev, sizeof(c64));
    z_gemm_hn(nev, nev, n, 1.0, X, wrk, 0.0, G);

    for (uint64_t i = 0; i < nev; i++) G[i + i * nev] -= (f64)sig[i];

    f64 err = z_nrm2(nev * nev, G);
    safe_free((void**)&G);
    return err;
}

/* ================================================================
 * Helper: Rayleigh quotient diagonality (indefinite)
 * For A*x = λ*B*x with X^H*B*X = diag(sig):
 *   X^H*A*X = diag(sig_i * λ_i)
 * Returns ||X^H*A*X - diag(sig_i * eigVal_i)||_F
 * Pass sig=NULL for standard case (diag(eigVal))
 * ================================================================ */
static f64 rayleigh_diag_d(uint64_t n, uint64_t nev, const f64 *A_mat,
                           const f64 *X, const f64 *eigVal,
                           const int8_t *sig) {
    f64 *AX = xcalloc(n * nev, sizeof(f64));
    f64 *G  = xcalloc(nev * nev, sizeof(f64));
    d_gemm_nn(n, nev, n, 1.0, A_mat, X, 0.0, AX);
    d_gemm_tn(nev, nev, n, 1.0, X, AX, 0.0, G);

    for (uint64_t i = 0; i < nev; i++) {
        f64 diag_val = sig ? (f64)sig[i] * eigVal[i] : eigVal[i];
        G[i + i*nev] -= diag_val;
    }

    f64 err = d_nrm2(nev * nev, G);
    safe_free((void**)&AX); safe_free((void**)&G);
    return err;
}

static f64 rayleigh_diag_z(uint64_t n, uint64_t nev, const c64 *A_mat,
                           const c64 *X, const f64 *eigVal,
                           const int8_t *sig) {
    c64 *AX = xcalloc(n * nev, sizeof(c64));
    c64 *G  = xcalloc(nev * nev, sizeof(c64));
    z_gemm_nn(n, nev, n, (c64)1, A_mat, X, (c64)0, AX);
    z_gemm_hn(nev, nev, n, (c64)1, X, AX, (c64)0, G);

    for (uint64_t i = 0; i < nev; i++) {
        f64 diag_val = sig ? (f64)sig[i] * eigVal[i] : eigVal[i];
        G[i + i*nev] -= diag_val;
    }

    f64 err = z_nrm2(nev * nev, G);
    safe_free((void**)&AX); safe_free((void**)&G);
    return err;
}

/* ================================================================
 * Category 1: Diagonal A + diagonal B (existing tests, converted)
 *
 * A = diag(1..6), B = diag(+1,+1,+1,-1,-1,-1)
 * eigenvalues: lambda_i = A_ii / B_ii = {1,2,3,-4,-5,-6}
 * sorted: {1,2,3,-4,-5,-6}, sig={+1,+1,+1,-1,-1,-1}
 * ================================================================ */

TEST(d_indef_rr_diag) {
    const uint64_t n = 6;

    diag_ctx_d_t *ctxA = xcalloc(1, sizeof(diag_ctx_d_t));
    ctxA->n = n;
    ctxA->diag = xcalloc(n, sizeof(f64));
    for (uint64_t i = 0; i < n; i++) ctxA->diag[i] = (f64)(i + 1);

    diag_ctx_d_t *ctxB = xcalloc(1, sizeof(diag_ctx_d_t));
    ctxB->n = n;
    ctxB->diag = xcalloc(n, sizeof(f64));
    for (uint64_t i = 0; i < n; i++)
        ctxB->diag[i] = (i < n / 2) ? 1.0 : -1.0;

    LinearOperator_d_t A = { .rows = n, .cols = n, .matvec = diag_matvec_d,
                             .ctx = (linop_ctx_t *)ctxA };
    LinearOperator_d_t B = { .rows = n, .cols = n, .matvec = diag_matvec_d,
                             .ctx = (linop_ctx_t *)ctxB };

    f64 *S    = xcalloc(n * n, sizeof(f64));
    f64 *Cx   = xcalloc(n * n, sizeof(f64));
    f64 *eigVal = xcalloc(n, sizeof(f64));
    int8_t *sig = xcalloc(n, sizeof(int8_t));
    f64 *wrk1 = xcalloc(n * n, sizeof(f64));
    f64 *wrk2 = xcalloc(n * n, sizeof(f64));
    f64 *wrk3 = xcalloc(n * n, sizeof(f64));
    f64 *wrk4 = xcalloc(n * n, sizeof(f64));

    for (uint64_t i = 0; i < n; i++) S[i + i * n] = 1.0;

    d_indefinite_rayleigh_ritz(n, n, S, Cx, eigVal, sig,
                                wrk1, wrk2, wrk3, wrk4, &A, &B);

    f64 expected_eig[] = {1.0, 2.0, 3.0, -4.0, -5.0, -6.0};
    int8_t expected_sig[] = {1, 1, 1, -1, -1, -1};

    f64 max_err = 0;
    int sig_ok = 1;
    for (uint64_t i = 0; i < n; i++) {
        f64 err = fabs(eigVal[i] - expected_eig[i]);
        if (err > max_err) max_err = err;
        if (sig[i] != expected_sig[i]) sig_ok = 0;
    }
    printf("err=%.2e ", max_err);
    ASSERT(max_err < TEST_TOL_D);
    ASSERT(1 == sig_ok);

    safe_free((void**)&S);    safe_free((void**)&Cx);
    safe_free((void**)&eigVal); safe_free((void**)&sig);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2);
    safe_free((void**)&wrk3); safe_free((void**)&wrk4);
    safe_free((void**)&ctxA->diag); safe_free((void**)&ctxA);
    safe_free((void**)&ctxB->diag); safe_free((void**)&ctxB);
}

TEST(s_indef_rr_diag) {
    const uint64_t n = 6;

    diag_ctx_s_t *ctxA = xcalloc(1, sizeof(diag_ctx_s_t));
    ctxA->n = n;
    ctxA->diag = xcalloc(n, sizeof(f32));
    for (uint64_t i = 0; i < n; i++) ctxA->diag[i] = (f32)(i + 1);

    diag_ctx_s_t *ctxB = xcalloc(1, sizeof(diag_ctx_s_t));
    ctxB->n = n;
    ctxB->diag = xcalloc(n, sizeof(f32));
    for (uint64_t i = 0; i < n; i++)
        ctxB->diag[i] = (i < n / 2) ? 1.0f : -1.0f;

    LinearOperator_s_t A = { .rows = n, .cols = n, .matvec = diag_matvec_s,
                             .ctx = (linop_ctx_t *)ctxA };
    LinearOperator_s_t B = { .rows = n, .cols = n, .matvec = diag_matvec_s,
                             .ctx = (linop_ctx_t *)ctxB };

    f32 *S    = xcalloc(n * n, sizeof(f32));
    f32 *Cx   = xcalloc(n * n, sizeof(f32));
    f32 *eigVal = xcalloc(n, sizeof(f32));
    int8_t *sig = xcalloc(n, sizeof(int8_t));
    f32 *wrk1 = xcalloc(n * n, sizeof(f32));
    f32 *wrk2 = xcalloc(n * n, sizeof(f32));
    f32 *wrk3 = xcalloc(n * n, sizeof(f32));
    f32 *wrk4 = xcalloc(n * n, sizeof(f32));

    for (uint64_t i = 0; i < n; i++) S[i + i * n] = 1.0f;

    s_indefinite_rayleigh_ritz(n, n, S, Cx, eigVal, sig,
                                wrk1, wrk2, wrk3, wrk4, &A, &B);

    f32 expected_eig[] = {1.0f, 2.0f, 3.0f, -4.0f, -5.0f, -6.0f};
    int8_t expected_sig[] = {1, 1, 1, -1, -1, -1};

    f32 max_err = 0;
    int sig_ok = 1;
    for (uint64_t i = 0; i < n; i++) {
        f32 err = fabsf(eigVal[i] - expected_eig[i]);
        if (err > max_err) max_err = err;
        if (sig[i] != expected_sig[i]) sig_ok = 0;
    }
    printf("err=%.2e ", max_err);
    ASSERT(max_err < TEST_TOL_S);
    ASSERT(1 == sig_ok);

    safe_free((void**)&S);    safe_free((void**)&Cx);
    safe_free((void**)&eigVal); safe_free((void**)&sig);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2);
    safe_free((void**)&wrk3); safe_free((void**)&wrk4);
    safe_free((void**)&ctxA->diag); safe_free((void**)&ctxA);
    safe_free((void**)&ctxB->diag); safe_free((void**)&ctxB);
}

TEST(z_indef_rr_diag) {
    const uint64_t n = 6;

    diag_ctx_z_t *ctxA = xcalloc(1, sizeof(diag_ctx_z_t));
    ctxA->n = n;
    ctxA->diag = xcalloc(n, sizeof(f64));
    for (uint64_t i = 0; i < n; i++) ctxA->diag[i] = (f64)(i + 1);

    diag_ctx_z_t *ctxB = xcalloc(1, sizeof(diag_ctx_z_t));
    ctxB->n = n;
    ctxB->diag = xcalloc(n, sizeof(f64));
    for (uint64_t i = 0; i < n; i++)
        ctxB->diag[i] = (i < n / 2) ? 1.0 : -1.0;

    LinearOperator_z_t A = { .rows = n, .cols = n, .matvec = diag_matvec_z,
                             .ctx = (linop_ctx_t *)ctxA };
    LinearOperator_z_t B = { .rows = n, .cols = n, .matvec = diag_matvec_z,
                             .ctx = (linop_ctx_t *)ctxB };

    c64 *S    = xcalloc(n * n, sizeof(c64));
    c64 *Cx   = xcalloc(n * n, sizeof(c64));
    f64 *eigVal = xcalloc(n, sizeof(f64));
    int8_t *sig = xcalloc(n, sizeof(int8_t));
    c64 *wrk1 = xcalloc(n * n, sizeof(c64));
    c64 *wrk2 = xcalloc(n * n, sizeof(c64));
    c64 *wrk3 = xcalloc(n * n, sizeof(c64));
    c64 *wrk4 = xcalloc(n * n, sizeof(c64));

    for (uint64_t i = 0; i < n; i++) S[i + i * n] = 1.0 + 0 * I;

    z_indefinite_rayleigh_ritz(n, n, S, Cx, eigVal, sig,
                                wrk1, wrk2, wrk3, wrk4, &A, &B);

    f64 expected_eig[] = {1.0, 2.0, 3.0, -4.0, -5.0, -6.0};
    int8_t expected_sig[] = {1, 1, 1, -1, -1, -1};

    f64 max_err = 0;
    int sig_ok = 1;
    for (uint64_t i = 0; i < n; i++) {
        f64 err = fabs(eigVal[i] - expected_eig[i]);
        if (err > max_err) max_err = err;
        if (sig[i] != expected_sig[i]) sig_ok = 0;
    }
    printf("err=%.2e ", max_err);
    ASSERT(max_err < TEST_TOL_D);
    ASSERT(1 == sig_ok);

    safe_free((void**)&S);    safe_free((void**)&Cx);
    safe_free((void**)&eigVal); safe_free((void**)&sig);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2);
    safe_free((void**)&wrk3); safe_free((void**)&wrk4);
    safe_free((void**)&ctxA->diag); safe_free((void**)&ctxA);
    safe_free((void**)&ctxB->diag); safe_free((void**)&ctxB);
}

TEST(c_indef_rr_diag) {
    const uint64_t n = 6;

    diag_ctx_c_t *ctxA = xcalloc(1, sizeof(diag_ctx_c_t));
    ctxA->n = n;
    ctxA->diag = xcalloc(n, sizeof(f32));
    for (uint64_t i = 0; i < n; i++) ctxA->diag[i] = (f32)(i + 1);

    diag_ctx_c_t *ctxB = xcalloc(1, sizeof(diag_ctx_c_t));
    ctxB->n = n;
    ctxB->diag = xcalloc(n, sizeof(f32));
    for (uint64_t i = 0; i < n; i++)
        ctxB->diag[i] = (i < n / 2) ? 1.0f : -1.0f;

    LinearOperator_c_t A = { .rows = n, .cols = n, .matvec = diag_matvec_c,
                             .ctx = (linop_ctx_t *)ctxA };
    LinearOperator_c_t B = { .rows = n, .cols = n, .matvec = diag_matvec_c,
                             .ctx = (linop_ctx_t *)ctxB };

    c32 *S    = xcalloc(n * n, sizeof(c32));
    c32 *Cx   = xcalloc(n * n, sizeof(c32));
    f32 *eigVal = xcalloc(n, sizeof(f32));
    int8_t *sig = xcalloc(n, sizeof(int8_t));
    c32 *wrk1 = xcalloc(n * n, sizeof(c32));
    c32 *wrk2 = xcalloc(n * n, sizeof(c32));
    c32 *wrk3 = xcalloc(n * n, sizeof(c32));
    c32 *wrk4 = xcalloc(n * n, sizeof(c32));

    for (uint64_t i = 0; i < n; i++) S[i + i * n] = 1.0f + 0 * I;

    c_indefinite_rayleigh_ritz(n, n, S, Cx, eigVal, sig,
                                wrk1, wrk2, wrk3, wrk4, &A, &B);

    f32 expected_eig[] = {1.0f, 2.0f, 3.0f, -4.0f, -5.0f, -6.0f};
    int8_t expected_sig[] = {1, 1, 1, -1, -1, -1};

    f32 max_err = 0;
    int sig_ok = 1;
    for (uint64_t i = 0; i < n; i++) {
        f32 err = fabsf(eigVal[i] - expected_eig[i]);
        if (err > max_err) max_err = err;
        if (sig[i] != expected_sig[i]) sig_ok = 0;
    }
    printf("err=%.2e ", max_err);
    ASSERT(max_err < TEST_TOL_S);
    ASSERT(1 == sig_ok);

    safe_free((void**)&S);    safe_free((void**)&Cx);
    safe_free((void**)&eigVal); safe_free((void**)&sig);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2);
    safe_free((void**)&wrk3); safe_free((void**)&wrk4);
    safe_free((void**)&ctxA->diag); safe_free((void**)&ctxA);
    safe_free((void**)&ctxB->diag); safe_free((void**)&ctxB);
}

/* ================================================================
 * Category 1 (continued): Modified Rayleigh-Ritz, diagonal A+B
 * ================================================================ */

TEST(d_indef_rr_modified_diag) {
    const uint64_t n = 6, nev = 3;
    const uint64_t mult = 2;
    const uint64_t sizeSub = mult * nev;

    diag_ctx_d_t *ctxA = xcalloc(1, sizeof(diag_ctx_d_t));
    ctxA->n = n;
    ctxA->diag = xcalloc(n, sizeof(f64));
    for (uint64_t i = 0; i < n; i++) ctxA->diag[i] = (f64)(i + 1);

    diag_ctx_d_t *ctxB = xcalloc(1, sizeof(diag_ctx_d_t));
    ctxB->n = n;
    ctxB->diag = xcalloc(n, sizeof(f64));
    for (uint64_t i = 0; i < n; i++)
        ctxB->diag[i] = (i < n / 2) ? 1.0 : -1.0;

    LinearOperator_d_t A = { .rows = n, .cols = n, .matvec = diag_matvec_d,
                             .ctx = (linop_ctx_t *)ctxA };
    LinearOperator_d_t B = { .rows = n, .cols = n, .matvec = diag_matvec_d,
                             .ctx = (linop_ctx_t *)ctxB };

    f64 *S    = xcalloc(n * sizeSub, sizeof(f64));
    for (uint64_t j = 0; j < sizeSub; j++) S[j + j * n] = 1.0;

    const uint64_t n_rem = (mult - 1) * nev;
    f64 *Cx   = xcalloc(sizeSub * nev, sizeof(f64));
    f64 *Cp   = xcalloc(sizeSub * n_rem, sizeof(f64));
    f64 *Cx_ortho = xcalloc(sizeSub * nev, sizeof(f64));
    int quality_flag = 0;
    f64 *eigVal = xcalloc(nev, sizeof(f64));
    int8_t *sig = xcalloc(sizeSub, sizeof(int8_t));
    f64 *wrk1 = xcalloc(sizeSub * sizeSub, sizeof(f64));
    f64 *wrk2 = xcalloc(n * sizeSub, sizeof(f64));
    f64 *wrk3 = xcalloc(sizeSub * sizeSub, sizeof(f64));
    f64 *wrk4 = xcalloc(sizeSub * sizeSub, sizeof(f64));

    d_indefinite_rayleigh_ritz_modified(n, nev, mult, 0, 0,
                                         S, wrk1, wrk2, wrk3, wrk4,
                                         Cx, Cp, Cx_ortho, eigVal, sig, &quality_flag, &A, &B);

    f64 expected_eig[] = {1.0, 2.0, 3.0};
    f64 max_err = 0;
    for (uint64_t i = 0; i < nev; i++) {
        f64 err = fabs(eigVal[i] - expected_eig[i]);
        if (err > max_err) max_err = err;
    }

    int sig_ok = 1;
    for (uint64_t i = 0; i < nev; i++)
        if (sig[i] != 1) sig_ok = 0;
    for (uint64_t i = nev; i < sizeSub; i++)
        if (sig[i] != -1) sig_ok = 0;

    printf("err=%.2e ", max_err);
    ASSERT(max_err < TEST_TOL_D);
    ASSERT(1 == sig_ok);

    safe_free((void**)&S);    safe_free((void**)&Cx);   safe_free((void**)&Cp);   safe_free((void**)&Cx_ortho);
    safe_free((void**)&eigVal); safe_free((void**)&sig);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2);
    safe_free((void**)&wrk3); safe_free((void**)&wrk4);
    safe_free((void**)&ctxA->diag); safe_free((void**)&ctxA);
    safe_free((void**)&ctxB->diag); safe_free((void**)&ctxB);
}

TEST(d_indef_rr_modified_diag_mult3) {
    const uint64_t n = 9, nev = 3;
    const uint64_t mult = 3;
    const uint64_t sizeSub = mult * nev;

    diag_ctx_d_t *ctxA = xcalloc(1, sizeof(diag_ctx_d_t));
    ctxA->n = n;
    ctxA->diag = xcalloc(n, sizeof(f64));
    for (uint64_t i = 0; i < n; i++) ctxA->diag[i] = (f64)(i + 1);

    diag_ctx_d_t *ctxB = xcalloc(1, sizeof(diag_ctx_d_t));
    ctxB->n = n;
    ctxB->diag = xcalloc(n, sizeof(f64));
    for (uint64_t i = 0; i < n; i++)
        ctxB->diag[i] = (i < 5) ? 1.0 : -1.0;

    LinearOperator_d_t A = { .rows = n, .cols = n, .matvec = diag_matvec_d,
                             .ctx = (linop_ctx_t *)ctxA };
    LinearOperator_d_t B = { .rows = n, .cols = n, .matvec = diag_matvec_d,
                             .ctx = (linop_ctx_t *)ctxB };

    f64 *S    = xcalloc(n * sizeSub, sizeof(f64));
    for (uint64_t j = 0; j < sizeSub; j++) S[j + j * n] = 1.0;

    const uint64_t n_rem = (mult - 1) * nev;
    f64 *Cx   = xcalloc(sizeSub * nev, sizeof(f64));
    f64 *Cp   = xcalloc(sizeSub * n_rem, sizeof(f64));
    f64 *Cx_ortho = xcalloc(sizeSub * nev, sizeof(f64));
    int quality_flag = 0;
    f64 *eigVal = xcalloc(nev, sizeof(f64));
    int8_t *sig = xcalloc(sizeSub, sizeof(int8_t));
    f64 *wrk1 = xcalloc(sizeSub * sizeSub, sizeof(f64));
    f64 *wrk2 = xcalloc(n * sizeSub, sizeof(f64));
    f64 *wrk3 = xcalloc(sizeSub * sizeSub, sizeof(f64));
    f64 *wrk4 = xcalloc(sizeSub * sizeSub, sizeof(f64));

    d_indefinite_rayleigh_ritz_modified(n, nev, mult, 0, 0,
                                         S, wrk1, wrk2, wrk3, wrk4,
                                         Cx, Cp, Cx_ortho, eigVal, sig, &quality_flag, &A, &B);

    f64 expected_eig[] = {1.0, 2.0, 3.0};
    f64 max_err = 0;
    for (uint64_t i = 0; i < nev; i++) {
        f64 err = fabs(eigVal[i] - expected_eig[i]);
        if (err > max_err) max_err = err;
    }
    printf("err=%.2e ", max_err);
    ASSERT(max_err < TEST_TOL_D);

    safe_free((void**)&S);    safe_free((void**)&Cx);   safe_free((void**)&Cp);   safe_free((void**)&Cx_ortho);
    safe_free((void**)&eigVal); safe_free((void**)&sig);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2);
    safe_free((void**)&wrk3); safe_free((void**)&wrk4);
    safe_free((void**)&ctxA->diag); safe_free((void**)&ctxA);
    safe_free((void**)&ctxB->diag); safe_free((void**)&ctxB);
}

TEST(z_indef_rr_modified_diag) {
    const uint64_t n = 6, nev = 3;
    const uint64_t mult = 2;
    const uint64_t sizeSub = mult * nev;

    diag_ctx_z_t *ctxA = xcalloc(1, sizeof(diag_ctx_z_t));
    ctxA->n = n;
    ctxA->diag = xcalloc(n, sizeof(f64));
    for (uint64_t i = 0; i < n; i++) ctxA->diag[i] = (f64)(i + 1);

    diag_ctx_z_t *ctxB = xcalloc(1, sizeof(diag_ctx_z_t));
    ctxB->n = n;
    ctxB->diag = xcalloc(n, sizeof(f64));
    for (uint64_t i = 0; i < n; i++)
        ctxB->diag[i] = (i < n / 2) ? 1.0 : -1.0;

    LinearOperator_z_t A = { .rows = n, .cols = n, .matvec = diag_matvec_z,
                             .ctx = (linop_ctx_t *)ctxA };
    LinearOperator_z_t B = { .rows = n, .cols = n, .matvec = diag_matvec_z,
                             .ctx = (linop_ctx_t *)ctxB };

    c64 *S    = xcalloc(n * sizeSub, sizeof(c64));
    for (uint64_t j = 0; j < sizeSub; j++) S[j + j * n] = 1.0 + 0 * I;

    const uint64_t n_rem = (mult - 1) * nev;
    c64 *Cx   = xcalloc(sizeSub * nev, sizeof(c64));
    c64 *Cp   = xcalloc(sizeSub * n_rem, sizeof(c64));
    c64 *Cx_ortho = xcalloc(sizeSub * nev, sizeof(c64));
    int quality_flag = 0;
    f64 *eigVal = xcalloc(nev, sizeof(f64));
    int8_t *sig = xcalloc(sizeSub, sizeof(int8_t));
    c64 *wrk1 = xcalloc(sizeSub * sizeSub, sizeof(c64));
    c64 *wrk2 = xcalloc(n * sizeSub, sizeof(c64));
    c64 *wrk3 = xcalloc(sizeSub * sizeSub, sizeof(c64));
    c64 *wrk4 = xcalloc(sizeSub * sizeSub, sizeof(c64));

    z_indefinite_rayleigh_ritz_modified(n, nev, mult, 0, 0,
                                         S, wrk1, wrk2, wrk3, wrk4,
                                         Cx, Cp, Cx_ortho, eigVal, sig, &quality_flag, &A, &B);

    f64 expected_eig[] = {1.0, 2.0, 3.0};
    f64 max_err = 0;
    for (uint64_t i = 0; i < nev; i++) {
        f64 err = fabs(eigVal[i] - expected_eig[i]);
        if (err > max_err) max_err = err;
    }
    printf("err=%.2e ", max_err);
    ASSERT(max_err < TEST_TOL_D);

    safe_free((void**)&S);    safe_free((void**)&Cx);   safe_free((void**)&Cp);   safe_free((void**)&Cx_ortho);
    safe_free((void**)&eigVal); safe_free((void**)&sig);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2);
    safe_free((void**)&wrk3); safe_free((void**)&wrk4);
    safe_free((void**)&ctxA->diag); safe_free((void**)&ctxA);
    safe_free((void**)&ctxB->diag); safe_free((void**)&ctxB);
}

/* ================================================================
 * Category 2: Diagonal A + block-permutation B (analytic eigenvalues)
 *
 * A = diag(1,2,3,4,5,6), B = blkdiag({{0,1},{1,0}} x 3)
 * Block (a,b): diag(a,b)*x = lambda*{{0,1},{1,0}}*x => lambda = +-sqrt(a*b)
 *   Block (1,2): lambda = +-sqrt(2)
 *   Block (3,4): lambda = +-sqrt(12) = +-2*sqrt(3)
 *   Block (5,6): lambda = +-sqrt(30)
 * Sorted: {sqrt(2), 2*sqrt(3), sqrt(30), -sqrt(2), -2*sqrt(3), -sqrt(30)}
 * sig = {+1,+1,+1,-1,-1,-1}
 * ================================================================ */

TEST(d_indef_rr_perm) {
    const uint64_t n = 6;

    diag_ctx_d_t *ctxA = xcalloc(1, sizeof(diag_ctx_d_t));
    ctxA->n = n;
    ctxA->diag = xcalloc(n, sizeof(f64));
    for (uint64_t i = 0; i < n; i++) ctxA->diag[i] = (f64)(i + 1);

    LinearOperator_d_t A = { .rows = n, .cols = n, .matvec = diag_matvec_d,
                             .ctx = (linop_ctx_t *)ctxA };
    LinearOperator_d_t *B = create_perm_B_d(n);

    f64 *S    = xcalloc(n * n, sizeof(f64));
    f64 *Cx   = xcalloc(n * n, sizeof(f64));
    f64 *eigVal = xcalloc(n, sizeof(f64));
    int8_t *sig = xcalloc(n, sizeof(int8_t));
    f64 *wrk1 = xcalloc(n * n, sizeof(f64));
    f64 *wrk2 = xcalloc(n * n, sizeof(f64));
    f64 *wrk3 = xcalloc(n * n, sizeof(f64));
    f64 *wrk4 = xcalloc(n * n, sizeof(f64));

    for (uint64_t i = 0; i < n; i++) S[i + i * n] = 1.0;

    d_indefinite_rayleigh_ritz(n, n, S, Cx, eigVal, sig,
                                wrk1, wrk2, wrk3, wrk4, &A, B);

    f64 expected_eig[] = {sqrt(2.0), 2.0*sqrt(3.0), sqrt(30.0),
                          -sqrt(2.0), -2.0*sqrt(3.0), -sqrt(30.0)};
    int8_t expected_sig[] = {1, 1, 1, -1, -1, -1};

    f64 max_err = 0;
    int sig_ok = 1;
    for (uint64_t i = 0; i < n; i++) {
        f64 err = fabs(eigVal[i] - expected_eig[i]);
        if (err > max_err) max_err = err;
        if (sig[i] != expected_sig[i]) sig_ok = 0;
    }

    /* B-sig-orthonormality: X = S*Cx */
    f64 *X = xcalloc(n * n, sizeof(f64));
    d_gemm_nn(n, n, n, 1.0, S, Cx, 0.0, X);
    f64 bso = B_sig_ortho_d(n, n, X, B, sig, wrk1);

    printf("err=%.2e bso=%.2e ", max_err, bso);
    ASSERT(max_err < TEST_TOL_D);
    ASSERT(1 == sig_ok);
    ASSERT(bso < TEST_TOL_D * n);

    safe_free((void**)&S);    safe_free((void**)&Cx);   safe_free((void**)&X);
    safe_free((void**)&eigVal); safe_free((void**)&sig);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2);
    safe_free((void**)&wrk3); safe_free((void**)&wrk4);
    safe_free((void**)&ctxA->diag); safe_free((void**)&ctxA);
    linop_destroy_d(&B);
}

TEST(z_indef_rr_perm) {
    const uint64_t n = 6;

    diag_ctx_z_t *ctxA = xcalloc(1, sizeof(diag_ctx_z_t));
    ctxA->n = n;
    ctxA->diag = xcalloc(n, sizeof(f64));
    for (uint64_t i = 0; i < n; i++) ctxA->diag[i] = (f64)(i + 1);

    LinearOperator_z_t A = { .rows = n, .cols = n, .matvec = diag_matvec_z,
                             .ctx = (linop_ctx_t *)ctxA };
    LinearOperator_z_t *B = create_perm_B_z(n);

    c64 *S    = xcalloc(n * n, sizeof(c64));
    c64 *Cx   = xcalloc(n * n, sizeof(c64));
    f64 *eigVal = xcalloc(n, sizeof(f64));
    int8_t *sig = xcalloc(n, sizeof(int8_t));
    c64 *wrk1 = xcalloc(n * n, sizeof(c64));
    c64 *wrk2 = xcalloc(n * n, sizeof(c64));
    c64 *wrk3 = xcalloc(n * n, sizeof(c64));
    c64 *wrk4 = xcalloc(n * n, sizeof(c64));

    for (uint64_t i = 0; i < n; i++) S[i + i * n] = 1.0 + 0 * I;

    z_indefinite_rayleigh_ritz(n, n, S, Cx, eigVal, sig,
                                wrk1, wrk2, wrk3, wrk4, &A, B);

    f64 expected_eig[] = {sqrt(2.0), 2.0*sqrt(3.0), sqrt(30.0),
                          -sqrt(2.0), -2.0*sqrt(3.0), -sqrt(30.0)};
    int8_t expected_sig[] = {1, 1, 1, -1, -1, -1};

    f64 max_err = 0;
    int sig_ok = 1;
    for (uint64_t i = 0; i < n; i++) {
        f64 err = fabs(eigVal[i] - expected_eig[i]);
        if (err > max_err) max_err = err;
        if (sig[i] != expected_sig[i]) sig_ok = 0;
    }

    c64 *X = xcalloc(n * n, sizeof(c64));
    z_gemm_nn(n, n, n, (c64)1, S, Cx, (c64)0, X);
    f64 bso = B_sig_ortho_z(n, n, X, B, sig, wrk1);

    printf("err=%.2e bso=%.2e ", max_err, bso);
    ASSERT(max_err < TEST_TOL_D);
    ASSERT(1 == sig_ok);
    ASSERT(bso < TEST_TOL_D * n);

    safe_free((void**)&S);    safe_free((void**)&Cx);   safe_free((void**)&X);
    safe_free((void**)&eigVal); safe_free((void**)&sig);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2);
    safe_free((void**)&wrk3); safe_free((void**)&wrk4);
    safe_free((void**)&ctxA->diag); safe_free((void**)&ctxA);
    linop_destroy_z(&B);
}

TEST(d_indef_rr_modified_perm) {
    const uint64_t n = 6, nev = 3, mult = 2;
    const uint64_t sizeSub = mult * nev;

    diag_ctx_d_t *ctxA = xcalloc(1, sizeof(diag_ctx_d_t));
    ctxA->n = n;
    ctxA->diag = xcalloc(n, sizeof(f64));
    for (uint64_t i = 0; i < n; i++) ctxA->diag[i] = (f64)(i + 1);

    LinearOperator_d_t A = { .rows = n, .cols = n, .matvec = diag_matvec_d,
                             .ctx = (linop_ctx_t *)ctxA };
    LinearOperator_d_t *B = create_perm_B_d(n);

    f64 *S = xcalloc(n * sizeSub, sizeof(f64));
    for (uint64_t j = 0; j < sizeSub; j++) S[j + j * n] = 1.0;

    const uint64_t n_rem = (mult - 1) * nev;
    f64 *Cx   = xcalloc(sizeSub * nev, sizeof(f64));
    f64 *Cp   = xcalloc(sizeSub * n_rem, sizeof(f64));
    f64 *Cx_ortho = xcalloc(sizeSub * nev, sizeof(f64));
    int quality_flag = 0;
    f64 *eigVal = xcalloc(nev, sizeof(f64));
    int8_t *sig = xcalloc(sizeSub, sizeof(int8_t));
    f64 *wrk1 = xcalloc(sizeSub * sizeSub, sizeof(f64));
    f64 *wrk2 = xcalloc(n * sizeSub, sizeof(f64));
    f64 *wrk3 = xcalloc(sizeSub * sizeSub, sizeof(f64));
    f64 *wrk4 = xcalloc(sizeSub * sizeSub, sizeof(f64));

    d_indefinite_rayleigh_ritz_modified(n, nev, mult, 0, 0,
                                         S, wrk1, wrk2, wrk3, wrk4,
                                         Cx, Cp, Cx_ortho, eigVal, sig, &quality_flag, &A, B);

    f64 expected_eig[] = {sqrt(2.0), 2.0*sqrt(3.0), sqrt(30.0)};
    f64 max_err = 0;
    for (uint64_t i = 0; i < nev; i++) {
        f64 err = fabs(eigVal[i] - expected_eig[i]);
        if (err > max_err) max_err = err;
    }
    printf("err=%.2e ", max_err);
    ASSERT(max_err < TEST_TOL_D);

    safe_free((void**)&S);    safe_free((void**)&Cx);   safe_free((void**)&Cp);   safe_free((void**)&Cx_ortho);
    safe_free((void**)&eigVal); safe_free((void**)&sig);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2);
    safe_free((void**)&wrk3); safe_free((void**)&wrk4);
    safe_free((void**)&ctxA->diag); safe_free((void**)&ctxA);
    linop_destroy_d(&B);
}

TEST(z_indef_rr_modified_perm) {
    const uint64_t n = 6, nev = 3, mult = 2;
    const uint64_t sizeSub = mult * nev;

    diag_ctx_z_t *ctxA = xcalloc(1, sizeof(diag_ctx_z_t));
    ctxA->n = n;
    ctxA->diag = xcalloc(n, sizeof(f64));
    for (uint64_t i = 0; i < n; i++) ctxA->diag[i] = (f64)(i + 1);

    LinearOperator_z_t A = { .rows = n, .cols = n, .matvec = diag_matvec_z,
                             .ctx = (linop_ctx_t *)ctxA };
    LinearOperator_z_t *B = create_perm_B_z(n);

    c64 *S = xcalloc(n * sizeSub, sizeof(c64));
    for (uint64_t j = 0; j < sizeSub; j++) S[j + j * n] = 1.0 + 0 * I;

    const uint64_t n_rem = (mult - 1) * nev;
    c64 *Cx   = xcalloc(sizeSub * nev, sizeof(c64));
    c64 *Cp   = xcalloc(sizeSub * n_rem, sizeof(c64));
    c64 *Cx_ortho = xcalloc(sizeSub * nev, sizeof(c64));
    int quality_flag = 0;
    f64 *eigVal = xcalloc(nev, sizeof(f64));
    int8_t *sig = xcalloc(sizeSub, sizeof(int8_t));
    c64 *wrk1 = xcalloc(sizeSub * sizeSub, sizeof(c64));
    c64 *wrk2 = xcalloc(n * sizeSub, sizeof(c64));
    c64 *wrk3 = xcalloc(sizeSub * sizeSub, sizeof(c64));
    c64 *wrk4 = xcalloc(sizeSub * sizeSub, sizeof(c64));

    z_indefinite_rayleigh_ritz_modified(n, nev, mult, 0, 0,
                                         S, wrk1, wrk2, wrk3, wrk4,
                                         Cx, Cp, Cx_ortho, eigVal, sig, &quality_flag, &A, B);

    f64 expected_eig[] = {sqrt(2.0), 2.0*sqrt(3.0), sqrt(30.0)};
    f64 max_err = 0;
    for (uint64_t i = 0; i < nev; i++) {
        f64 err = fabs(eigVal[i] - expected_eig[i]);
        if (err > max_err) max_err = err;
    }
    printf("err=%.2e ", max_err);
    ASSERT(max_err < TEST_TOL_D);

    safe_free((void**)&S);    safe_free((void**)&Cx);   safe_free((void**)&Cp);   safe_free((void**)&Cx_ortho);
    safe_free((void**)&eigVal); safe_free((void**)&sig);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2);
    safe_free((void**)&wrk3); safe_free((void**)&wrk4);
    safe_free((void**)&ctxA->diag); safe_free((void**)&ctxA);
    linop_destroy_z(&B);
}

/* ================================================================
 * Category 3: Dense A + diagonal indefinite B (structural checks)
 *
 * A = A6x6, B = diag(+1,+1,+1,-1,-1,-1)
 * No analytic eigenvalues — verify structural properties:
 *   1. ||X^H*A*X - diag(eigVal)||_F < tol
 *   2. ||X^H*B*X - diag(sig)||_F < tol
 *   3. Positive eigvals ascending, negative descending
 * ================================================================ */

TEST(d_indef_rr_dense) {
    const uint64_t n = 6;

    dense_ctx_d_t *ctxA = xcalloc(1, sizeof(dense_ctx_d_t));
    ctxA->n = n;
    ctxA->A = xcalloc(n * n, sizeof(f64));
    memcpy(ctxA->A, A6x6, n * n * sizeof(f64));

    diag_ctx_d_t *ctxB = xcalloc(1, sizeof(diag_ctx_d_t));
    ctxB->n = n;
    ctxB->diag = xcalloc(n, sizeof(f64));
    for (uint64_t i = 0; i < n; i++)
        ctxB->diag[i] = (i < n / 2) ? 1.0 : -1.0;

    LinearOperator_d_t A = { .rows = n, .cols = n, .matvec = dense_matvec_d,
                             .ctx = (linop_ctx_t *)ctxA };
    LinearOperator_d_t B = { .rows = n, .cols = n, .matvec = diag_matvec_d,
                             .ctx = (linop_ctx_t *)ctxB };

    f64 *S    = xcalloc(n * n, sizeof(f64));
    f64 *Cx   = xcalloc(n * n, sizeof(f64));
    f64 *eigVal = xcalloc(n, sizeof(f64));
    int8_t *sig = xcalloc(n, sizeof(int8_t));
    f64 *wrk1 = xcalloc(n * n, sizeof(f64));
    f64 *wrk2 = xcalloc(n * n, sizeof(f64));
    f64 *wrk3 = xcalloc(n * n, sizeof(f64));
    f64 *wrk4 = xcalloc(n * n, sizeof(f64));

    for (uint64_t i = 0; i < n; i++) S[i + i * n] = 1.0;

    d_indefinite_rayleigh_ritz(n, n, S, Cx, eigVal, sig,
                                wrk1, wrk2, wrk3, wrk4, &A, &B);

    /* Reconstruct X = S * Cx */
    f64 *X = xcalloc(n * n, sizeof(f64));
    d_gemm_nn(n, n, n, 1.0, S, Cx, 0.0, X);

    /* Rayleigh quotient: X^T*A*X = diag(eigVal) */
    f64 rq = rayleigh_diag_d(n, n, ctxA->A, X, eigVal, sig);

    /* B-sig-orthonormality */
    f64 bso = B_sig_ortho_d(n, n, X, &B, sig, wrk1);

    /* Sort check: positive ascending, negative descending */
    int sort_ok = 1;
    uint64_t n_pos = 0;
    for (uint64_t i = 0; i < n; i++) if (1 == sig[i]) n_pos++;
    for (uint64_t i = 1; i < n_pos; i++)
        if (eigVal[i] < eigVal[i-1]) sort_ok = 0;
    for (uint64_t i = n_pos + 1; i < n; i++)
        if (eigVal[i] > eigVal[i-1]) sort_ok = 0;

    printf("rq=%.2e bso=%.2e ", rq, bso);
    ASSERT(rq < 1e-8);
    ASSERT(bso < 1e-8);
    ASSERT(1 == sort_ok);

    safe_free((void**)&S);    safe_free((void**)&Cx);   safe_free((void**)&X);
    safe_free((void**)&eigVal); safe_free((void**)&sig);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2);
    safe_free((void**)&wrk3); safe_free((void**)&wrk4);
    safe_free((void**)&ctxA->A); safe_free((void**)&ctxA);
    safe_free((void**)&ctxB->diag); safe_free((void**)&ctxB);
}

TEST(z_indef_rr_dense) {
    const uint64_t n = 6;

    dense_ctx_z_t *ctxA = xcalloc(1, sizeof(dense_ctx_z_t));
    ctxA->n = n;
    ctxA->A = xcalloc(n * n, sizeof(c64));
    for (uint64_t i = 0; i < n * n; i++) ctxA->A[i] = A6x6[i] + 0*I;

    diag_ctx_z_t *ctxB = xcalloc(1, sizeof(diag_ctx_z_t));
    ctxB->n = n;
    ctxB->diag = xcalloc(n, sizeof(f64));
    for (uint64_t i = 0; i < n; i++)
        ctxB->diag[i] = (i < n / 2) ? 1.0 : -1.0;

    LinearOperator_z_t A = { .rows = n, .cols = n, .matvec = dense_matvec_z,
                             .ctx = (linop_ctx_t *)ctxA };
    LinearOperator_z_t B = { .rows = n, .cols = n, .matvec = diag_matvec_z,
                             .ctx = (linop_ctx_t *)ctxB };

    c64 *S    = xcalloc(n * n, sizeof(c64));
    c64 *Cx   = xcalloc(n * n, sizeof(c64));
    f64 *eigVal = xcalloc(n, sizeof(f64));
    int8_t *sig = xcalloc(n, sizeof(int8_t));
    c64 *wrk1 = xcalloc(n * n, sizeof(c64));
    c64 *wrk2 = xcalloc(n * n, sizeof(c64));
    c64 *wrk3 = xcalloc(n * n, sizeof(c64));
    c64 *wrk4 = xcalloc(n * n, sizeof(c64));

    for (uint64_t i = 0; i < n; i++) S[i + i * n] = 1.0 + 0 * I;

    z_indefinite_rayleigh_ritz(n, n, S, Cx, eigVal, sig,
                                wrk1, wrk2, wrk3, wrk4, &A, &B);

    c64 *X = xcalloc(n * n, sizeof(c64));
    z_gemm_nn(n, n, n, (c64)1, S, Cx, (c64)0, X);

    f64 rq = rayleigh_diag_z(n, n, ctxA->A, X, eigVal, sig);
    f64 bso = B_sig_ortho_z(n, n, X, &B, sig, wrk1);

    int sort_ok = 1;
    uint64_t n_pos = 0;
    for (uint64_t i = 0; i < n; i++) if (1 == sig[i]) n_pos++;
    for (uint64_t i = 1; i < n_pos; i++)
        if (eigVal[i] < eigVal[i-1]) sort_ok = 0;
    for (uint64_t i = n_pos + 1; i < n; i++)
        if (eigVal[i] > eigVal[i-1]) sort_ok = 0;

    printf("rq=%.2e bso=%.2e ", rq, bso);
    ASSERT(rq < 1e-8);
    ASSERT(bso < 1e-8);
    ASSERT(1 == sort_ok);

    safe_free((void**)&S);    safe_free((void**)&Cx);   safe_free((void**)&X);
    safe_free((void**)&eigVal); safe_free((void**)&sig);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2);
    safe_free((void**)&wrk3); safe_free((void**)&wrk4);
    safe_free((void**)&ctxA->A); safe_free((void**)&ctxA);
    safe_free((void**)&ctxB->diag); safe_free((void**)&ctxB);
}

TEST(d_indef_rr_modified_dense) {
    const uint64_t n = 6, nev = 3, mult = 2;
    const uint64_t sizeSub = mult * nev;

    dense_ctx_d_t *ctxA = xcalloc(1, sizeof(dense_ctx_d_t));
    ctxA->n = n;
    ctxA->A = xcalloc(n * n, sizeof(f64));
    memcpy(ctxA->A, A6x6, n * n * sizeof(f64));

    diag_ctx_d_t *ctxB = xcalloc(1, sizeof(diag_ctx_d_t));
    ctxB->n = n;
    ctxB->diag = xcalloc(n, sizeof(f64));
    for (uint64_t i = 0; i < n; i++)
        ctxB->diag[i] = (i < n / 2) ? 1.0 : -1.0;

    LinearOperator_d_t A = { .rows = n, .cols = n, .matvec = dense_matvec_d,
                             .ctx = (linop_ctx_t *)ctxA };
    LinearOperator_d_t B = { .rows = n, .cols = n, .matvec = diag_matvec_d,
                             .ctx = (linop_ctx_t *)ctxB };

    f64 *S = xcalloc(n * sizeSub, sizeof(f64));
    for (uint64_t j = 0; j < sizeSub; j++) S[j + j * n] = 1.0;

    const uint64_t n_rem = (mult - 1) * nev;
    f64 *Cx   = xcalloc(sizeSub * nev, sizeof(f64));
    f64 *Cp   = xcalloc(sizeSub * n_rem, sizeof(f64));
    f64 *Cx_ortho = xcalloc(sizeSub * nev, sizeof(f64));
    int quality_flag = 0;
    f64 *eigVal = xcalloc(nev, sizeof(f64));
    int8_t *sig = xcalloc(sizeSub, sizeof(int8_t));
    f64 *wrk1 = xcalloc(sizeSub * sizeSub, sizeof(f64));
    f64 *wrk2 = xcalloc(n * sizeSub, sizeof(f64));
    f64 *wrk3 = xcalloc(sizeSub * sizeSub, sizeof(f64));
    f64 *wrk4 = xcalloc(sizeSub * sizeSub, sizeof(f64));

    d_indefinite_rayleigh_ritz_modified(n, nev, mult, 0, 0,
                                         S, wrk1, wrk2, wrk3, wrk4,
                                         Cx, Cp, Cx_ortho, eigVal, sig, &quality_flag, &A, &B);

    /* Reconstruct X = S * Cx */
    f64 *X = xcalloc(n * nev, sizeof(f64));
    d_gemm_nn(n, nev, sizeSub, 1.0, S, Cx, 0.0, X);

    f64 rq = rayleigh_diag_d(n, nev, ctxA->A, X, eigVal, sig);

    /* Positive eigenvalues should be ascending */
    int sort_ok = 1;
    for (uint64_t i = 1; i < nev; i++)
        if (eigVal[i] < eigVal[i-1]) sort_ok = 0;

    printf("rq=%.2e ", rq);
    ASSERT(rq < 1e-8);
    ASSERT(1 == sort_ok);

    safe_free((void**)&S);    safe_free((void**)&Cx);   safe_free((void**)&Cp);   safe_free((void**)&Cx_ortho);
    safe_free((void**)&eigVal); safe_free((void**)&sig); safe_free((void**)&X);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2);
    safe_free((void**)&wrk3); safe_free((void**)&wrk4);
    safe_free((void**)&ctxA->A); safe_free((void**)&ctxA);
    safe_free((void**)&ctxB->diag); safe_free((void**)&ctxB);
}

TEST(z_indef_rr_modified_dense) {
    const uint64_t n = 6, nev = 3, mult = 2;
    const uint64_t sizeSub = mult * nev;

    dense_ctx_z_t *ctxA = xcalloc(1, sizeof(dense_ctx_z_t));
    ctxA->n = n;
    ctxA->A = xcalloc(n * n, sizeof(c64));
    for (uint64_t i = 0; i < n * n; i++) ctxA->A[i] = A6x6[i] + 0*I;

    diag_ctx_z_t *ctxB = xcalloc(1, sizeof(diag_ctx_z_t));
    ctxB->n = n;
    ctxB->diag = xcalloc(n, sizeof(f64));
    for (uint64_t i = 0; i < n; i++)
        ctxB->diag[i] = (i < n / 2) ? 1.0 : -1.0;

    LinearOperator_z_t A = { .rows = n, .cols = n, .matvec = dense_matvec_z,
                             .ctx = (linop_ctx_t *)ctxA };
    LinearOperator_z_t B = { .rows = n, .cols = n, .matvec = diag_matvec_z,
                             .ctx = (linop_ctx_t *)ctxB };

    c64 *S = xcalloc(n * sizeSub, sizeof(c64));
    for (uint64_t j = 0; j < sizeSub; j++) S[j + j * n] = 1.0 + 0 * I;

    const uint64_t n_rem = (mult - 1) * nev;
    c64 *Cx   = xcalloc(sizeSub * nev, sizeof(c64));
    c64 *Cp   = xcalloc(sizeSub * n_rem, sizeof(c64));
    c64 *Cx_ortho = xcalloc(sizeSub * nev, sizeof(c64));
    int quality_flag = 0;
    f64 *eigVal = xcalloc(nev, sizeof(f64));
    int8_t *sig = xcalloc(sizeSub, sizeof(int8_t));
    c64 *wrk1 = xcalloc(sizeSub * sizeSub, sizeof(c64));
    c64 *wrk2 = xcalloc(n * sizeSub, sizeof(c64));
    c64 *wrk3 = xcalloc(sizeSub * sizeSub, sizeof(c64));
    c64 *wrk4 = xcalloc(sizeSub * sizeSub, sizeof(c64));

    z_indefinite_rayleigh_ritz_modified(n, nev, mult, 0, 0,
                                         S, wrk1, wrk2, wrk3, wrk4,
                                         Cx, Cp, Cx_ortho, eigVal, sig, &quality_flag, &A, &B);

    c64 *X = xcalloc(n * nev, sizeof(c64));
    z_gemm_nn(n, nev, sizeSub, (c64)1, S, Cx, (c64)0, X);

    f64 rq = rayleigh_diag_z(n, nev, ctxA->A, X, eigVal, sig);

    int sort_ok = 1;
    for (uint64_t i = 1; i < nev; i++)
        if (eigVal[i] < eigVal[i-1]) sort_ok = 0;

    printf("rq=%.2e ", rq);
    ASSERT(rq < 1e-8);
    ASSERT(1 == sort_ok);

    safe_free((void**)&S);    safe_free((void**)&Cx);   safe_free((void**)&Cp);   safe_free((void**)&Cx_ortho);
    safe_free((void**)&eigVal); safe_free((void**)&sig); safe_free((void**)&X);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2);
    safe_free((void**)&wrk3); safe_free((void**)&wrk4);
    safe_free((void**)&ctxA->A); safe_free((void**)&ctxA);
    safe_free((void**)&ctxB->diag); safe_free((void**)&ctxB);
}

/* ================================================================
 * Category 4: Dense A + block-permutation B (structural checks)
 * ================================================================ */

TEST(d_indef_rr_dense_perm) {
    const uint64_t n = 6;

    dense_ctx_d_t *ctxA = xcalloc(1, sizeof(dense_ctx_d_t));
    ctxA->n = n;
    ctxA->A = xcalloc(n * n, sizeof(f64));
    memcpy(ctxA->A, A6x6, n * n * sizeof(f64));

    LinearOperator_d_t A = { .rows = n, .cols = n, .matvec = dense_matvec_d,
                             .ctx = (linop_ctx_t *)ctxA };
    LinearOperator_d_t *B = create_perm_B_d(n);

    f64 *S    = xcalloc(n * n, sizeof(f64));
    f64 *Cx   = xcalloc(n * n, sizeof(f64));
    f64 *eigVal = xcalloc(n, sizeof(f64));
    int8_t *sig = xcalloc(n, sizeof(int8_t));
    f64 *wrk1 = xcalloc(n * n, sizeof(f64));
    f64 *wrk2 = xcalloc(n * n, sizeof(f64));
    f64 *wrk3 = xcalloc(n * n, sizeof(f64));
    f64 *wrk4 = xcalloc(n * n, sizeof(f64));

    for (uint64_t i = 0; i < n; i++) S[i + i * n] = 1.0;

    d_indefinite_rayleigh_ritz(n, n, S, Cx, eigVal, sig,
                                wrk1, wrk2, wrk3, wrk4, &A, B);

    f64 *X = xcalloc(n * n, sizeof(f64));
    d_gemm_nn(n, n, n, 1.0, S, Cx, 0.0, X);

    f64 rq = rayleigh_diag_d(n, n, ctxA->A, X, eigVal, sig);
    f64 bso = B_sig_ortho_d(n, n, X, B, sig, wrk1);

    int sort_ok = 1;
    uint64_t n_pos = 0;
    for (uint64_t i = 0; i < n; i++) if (1 == sig[i]) n_pos++;
    for (uint64_t i = 1; i < n_pos; i++)
        if (eigVal[i] < eigVal[i-1]) sort_ok = 0;
    for (uint64_t i = n_pos + 1; i < n; i++)
        if (eigVal[i] > eigVal[i-1]) sort_ok = 0;

    printf("rq=%.2e bso=%.2e ", rq, bso);
    ASSERT(rq < 1e-8);
    ASSERT(bso < 1e-8);
    ASSERT(1 == sort_ok);

    safe_free((void**)&S);    safe_free((void**)&Cx);   safe_free((void**)&X);
    safe_free((void**)&eigVal); safe_free((void**)&sig);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2);
    safe_free((void**)&wrk3); safe_free((void**)&wrk4);
    safe_free((void**)&ctxA->A); safe_free((void**)&ctxA);
    linop_destroy_d(&B);
}

TEST(z_indef_rr_dense_perm) {
    const uint64_t n = 6;

    dense_ctx_z_t *ctxA = xcalloc(1, sizeof(dense_ctx_z_t));
    ctxA->n = n;
    ctxA->A = xcalloc(n * n, sizeof(c64));
    for (uint64_t i = 0; i < n * n; i++) ctxA->A[i] = A6x6[i] + 0*I;

    LinearOperator_z_t A = { .rows = n, .cols = n, .matvec = dense_matvec_z,
                             .ctx = (linop_ctx_t *)ctxA };
    LinearOperator_z_t *B = create_perm_B_z(n);

    c64 *S    = xcalloc(n * n, sizeof(c64));
    c64 *Cx   = xcalloc(n * n, sizeof(c64));
    f64 *eigVal = xcalloc(n, sizeof(f64));
    int8_t *sig = xcalloc(n, sizeof(int8_t));
    c64 *wrk1 = xcalloc(n * n, sizeof(c64));
    c64 *wrk2 = xcalloc(n * n, sizeof(c64));
    c64 *wrk3 = xcalloc(n * n, sizeof(c64));
    c64 *wrk4 = xcalloc(n * n, sizeof(c64));

    for (uint64_t i = 0; i < n; i++) S[i + i * n] = 1.0 + 0 * I;

    z_indefinite_rayleigh_ritz(n, n, S, Cx, eigVal, sig,
                                wrk1, wrk2, wrk3, wrk4, &A, B);

    c64 *X = xcalloc(n * n, sizeof(c64));
    z_gemm_nn(n, n, n, (c64)1, S, Cx, (c64)0, X);

    f64 rq = rayleigh_diag_z(n, n, ctxA->A, X, eigVal, sig);
    f64 bso = B_sig_ortho_z(n, n, X, B, sig, wrk1);

    int sort_ok = 1;
    uint64_t n_pos = 0;
    for (uint64_t i = 0; i < n; i++) if (1 == sig[i]) n_pos++;
    for (uint64_t i = 1; i < n_pos; i++)
        if (eigVal[i] < eigVal[i-1]) sort_ok = 0;
    for (uint64_t i = n_pos + 1; i < n; i++)
        if (eigVal[i] > eigVal[i-1]) sort_ok = 0;

    printf("rq=%.2e bso=%.2e ", rq, bso);
    ASSERT(rq < 1e-8);
    ASSERT(bso < 1e-8);
    ASSERT(1 == sort_ok);

    safe_free((void**)&S);    safe_free((void**)&Cx);   safe_free((void**)&X);
    safe_free((void**)&eigVal); safe_free((void**)&sig);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2);
    safe_free((void**)&wrk3); safe_free((void**)&wrk4);
    safe_free((void**)&ctxA->A); safe_free((void**)&ctxA);
    linop_destroy_z(&B);
}

/* ================================================================ */
int main(void) {
    srand((unsigned)time(NULL));

    printf("Diagonal A + diagonal B:\n");
    RUN(d_indef_rr_diag);
    RUN(s_indef_rr_diag);
    RUN(z_indef_rr_diag);
    RUN(c_indef_rr_diag);
    RUN(d_indef_rr_modified_diag);
    RUN(d_indef_rr_modified_diag_mult3);
    RUN(z_indef_rr_modified_diag);

    printf("\nDiagonal A + permutation B:\n");
    RUN(d_indef_rr_perm);
    RUN(z_indef_rr_perm);
    RUN(d_indef_rr_modified_perm);
    RUN(z_indef_rr_modified_perm);

    printf("\nDense A + diagonal B:\n");
    RUN(d_indef_rr_dense);
    RUN(z_indef_rr_dense);
    RUN(d_indef_rr_modified_dense);
    RUN(z_indef_rr_modified_dense);

    printf("\nDense A + permutation B:\n");
    RUN(d_indef_rr_dense_perm);
    RUN(z_indef_rr_dense_perm);

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}
