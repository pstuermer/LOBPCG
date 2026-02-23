/**
 * @file test_ortho_indefinite.c
 * @brief Unit tests for indefinite B-orthogonalization
 *
 * Tests ortho_indefinite with an indefinite B matrix (diagonal with ±1).
 * Verifies:
 *   1. ||V^H*B*U||_F < tol (U is B-orthogonal to V)
 *   2. ||U^H*B*U - I||_F < tol (U is B-orthonormal)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <complex.h>
#include "lobpcg/types.h"
#include "lobpcg.h"
#include "lobpcg/blas_wrapper.h"
#include "lobpcg/linop.h"
#include "lobpcg/memory.h"

#define TOL_F32 1e-4
#define TOL_F64 1e-10

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

/* ====================================================================
 * Indefinite diagonal B operator
 *
 * B = diag(1, 1, ..., 1, -1, -1, ..., -1)
 *     (n_pos ones followed by n_neg negative ones)
 * ==================================================================== */

typedef struct {
    uint64_t n;
    uint64_t n_pos;  /* Number of positive entries */
} indef_diag_ctx_t;

static void indef_diag_matvec_d(const LinearOperator_d_t *op, f64 *x, f64 *y) {
    indef_diag_ctx_t *ctx = (indef_diag_ctx_t *)op->ctx->data;
    for (uint64_t i = 0; i < ctx->n; i++) {
        y[i] = (i < ctx->n_pos) ? x[i] : -x[i];
    }
}

static void indef_diag_matvec_z(const LinearOperator_z_t *op, c64 *x, c64 *y) {
    indef_diag_ctx_t *ctx = (indef_diag_ctx_t *)op->ctx->data;
    for (uint64_t i = 0; i < ctx->n; i++) {
        y[i] = (i < ctx->n_pos) ? x[i] : -x[i];
    }
}

static void indef_diag_cleanup(linop_ctx_t *ctx) {
    if (ctx && ctx->data) safe_free((void**)&ctx->data);
    if (ctx) safe_free((void**)&ctx);
}

static LinearOperator_d_t *create_indef_B_d(uint64_t n, uint64_t n_pos) {
    linop_ctx_t *ctx = xcalloc(1,sizeof(linop_ctx_t));
    indef_diag_ctx_t *data = xcalloc(1,sizeof(indef_diag_ctx_t));
    data->n = n;
    data->n_pos = n_pos;
    ctx->data = data;
    ctx->data_size = sizeof(indef_diag_ctx_t);
    return linop_create_d(n, n, indef_diag_matvec_d, indef_diag_cleanup, ctx);
}

static LinearOperator_z_t *create_indef_B_z(uint64_t n, uint64_t n_pos) {
    linop_ctx_t *ctx = xcalloc(1,sizeof(linop_ctx_t));
    indef_diag_ctx_t *data = xcalloc(1,sizeof(indef_diag_ctx_t));
    data->n = n;
    data->n_pos = n_pos;
    ctx->data = data;
    ctx->data_size = sizeof(indef_diag_ctx_t);
    return linop_create_z(n, n, indef_diag_matvec_z, indef_diag_cleanup, ctx);
}

/**
 * Compute B-orthogonality error ||V^H*B*U||_F
 */
static f64 B_cross_error_d(uint64_t m, uint64_t n_u, uint64_t n_v,
                           const f64 *U, const f64 *V,
                           LinearOperator_d_t *B, f64 *wrk) {
    for (uint64_t j = 0; j < n_u; j++)
        B->matvec(B, (f64*)&U[j * m], &wrk[j * m]);

    f64 *G = xcalloc(n_v * n_u, sizeof(f64));
    d_gemm_tn(n_v, n_u, m, 1.0, V, wrk, 0.0, G);

    f64 err = d_nrm2(n_v * n_u, G);
    safe_free((void**)&G);
    return err;
}

static f64 B_cross_error_z(uint64_t m, uint64_t n_u, uint64_t n_v,
                           const c64 *U, const c64 *V,
                           LinearOperator_z_t *B, c64 *wrk) {
    for (uint64_t j = 0; j < n_u; j++)
        B->matvec(B, (c64*)&U[j * m], &wrk[j * m]);

    c64 *G = xcalloc(n_v * n_u, sizeof(c64));
    z_gemm_hn(n_v, n_u, m, 1.0, V, wrk, 0.0, G);

    f64 err = z_nrm2(n_v * n_u, G);
    safe_free((void**)&G);
    return err;
}

/**
 * Compute B-orthonormality error for INDEFINITE B
 *
 * For indefinite B, U^H*B*U should be a SIGNATURE matrix (diagonal with ±1),
 * not necessarily the identity. We check:
 *   - Off-diagonal elements are near zero
 *   - Diagonal elements are near ±1
 *
 * Returns: max(||off-diag||_F, max_i ||G_ii| - 1|)
 */
static f64 B_norm_error_d(uint64_t m, uint64_t n,
                               const f64 *U, LinearOperator_d_t *B, f64 *wrk) {
    /* BU = B*U */
    for (uint64_t j = 0; j < n; j++)
        B->matvec(B, (f64*)&U[j * m], &wrk[j * m]);

    /* G = U^T * BU */
    f64 *G = xcalloc(n * n, sizeof(f64));
    d_gemm_tn(n, n, m, 1.0, U, wrk, 0.0, G);

    /* Check off-diagonal elements should be zero */
    f64 off_diag_err = 0;
    for (uint64_t j = 0; j < n; j++) {
        for (uint64_t i = 0; i < n; i++) {
            if (i != j) {
                off_diag_err += G[i + j * n] * G[i + j * n];
            }
        }
    }
    off_diag_err = sqrt(off_diag_err);

    /* Check diagonal elements should be ±1 */
    f64 diag_err = 0;
    for (uint64_t i = 0; i < n; i++) {
        f64 d = fabs(G[i + i * n]) - 1.0;  /* |G_ii| should be 1 */
        if (fabs(d) > diag_err) diag_err = fabs(d);
    }

    safe_free((void**)&G);
    return (off_diag_err > diag_err) ? off_diag_err : diag_err;
}

static f64 B_norm_error_z(uint64_t m, uint64_t n,
                               const c64 *U, LinearOperator_z_t *B, c64 *wrk) {
    /* BU = B*U */
    for (uint64_t j = 0; j < n; j++)
        B->matvec(B, (c64*)&U[j * m], &wrk[j * m]);

    /* G = U^H * BU */
    c64 *G = xcalloc(n * n, sizeof(c64));
    z_gemm_hn(n, n, m, 1.0, U, wrk, 0.0, G);

    /* Check off-diagonal elements should be zero */
    f64 off_diag_err = 0;
    for (uint64_t j = 0; j < n; j++) {
        for (uint64_t i = 0; i < n; i++) {
            if (i != j) {
                off_diag_err += cabs(G[i + j * n]) * cabs(G[i + j * n]);
            }
        }
    }
    off_diag_err = sqrt(off_diag_err);

    /* Check diagonal elements should be ±1 (real for Hermitian) */
    f64 diag_err = 0;
    for (uint64_t i = 0; i < n; i++) {
        f64 d = fabs(creal(G[i + i * n])) - 1.0;  /* |Re(G_ii)| should be 1 */
        f64 im = fabs(cimag(G[i + i * n]));       /* Im(G_ii) should be 0 */
        f64 err = (fabs(d) > im) ? fabs(d) : im;
        if (err > diag_err) diag_err = err;
    }

    safe_free((void**)&G);
    return (off_diag_err > diag_err) ? off_diag_err : diag_err;
}

/* ====================================================================
 * Helper functions for B=NULL tests
 * ==================================================================== */

/* ||V^H*U||_F;  wrk: >= n_v*n_u elements */
static f64 cross_error_d(uint64_t m, uint64_t n_u, uint64_t n_v,
                          const f64 *U, const f64 *V, f64 *wrk) {
    d_gemm_tn(n_v, n_u, m, 1.0, V, U, 0.0, wrk);
    return d_nrm2(n_v * n_u, wrk);
}

/* ||U^H*U - I||_F;  wrk: >= n*n elements */
static f64 norm_error_d(uint64_t m, uint64_t n, const f64 *U, f64 *wrk) {
    d_gemm_tn(n, n, m, 1.0, U, U, 0.0, wrk);
    for (uint64_t i = 0; i < n; i++) wrk[i + i*n] -= 1.0;
    return d_nrm2(n * n, wrk);
}

static f64 cross_error_z(uint64_t m, uint64_t n_u, uint64_t n_v,
                          const c64 *U, const c64 *V, c64 *wrk) {
    z_gemm_hn(n_v, n_u, m, 1.0, V, U, 0.0, wrk);
    return z_nrm2(n_v * n_u, wrk);
}

static f64 norm_error_z(uint64_t m, uint64_t n, const c64 *U, c64 *wrk) {
    z_gemm_hn(n, n, m, 1.0, U, U, 0.0, wrk);
    for (uint64_t i = 0; i < n; i++) wrk[i + i*n] -= 1.0;
    return z_nrm2(n * n, wrk);
}

/* ====================================================================
 * Double precision tests
 * ==================================================================== */

TEST(d_ortho_indefinite_basic) {
    /*
     * Test basic indefinite orthogonalization:
     * - Create indefinite B = diag(1,1,1,...,-1,-1,-1)
     * - Generate random V, B-orthonormalize it
     * - Generate random U
     * - Call ortho_indefinite
     * - Verify U is B-orthogonal to V and B-orthonormal
     */
    const uint64_t m = 100;    /* Problem size */
    const uint64_t n_v = 5;    /* External basis size */
    const uint64_t n_u = 8;    /* Candidate basis size */
    const uint64_t n_pos = 60; /* Positive entries in B */

    /* Create indefinite B */
    LinearOperator_d_t *B = create_indef_B_d(m, n_pos);

    /* Allocate matrices */
    f64 *V = xcalloc(m * n_v, sizeof(f64));
    f64 *U = xcalloc(m * n_u, sizeof(f64));
    uint64_t max_cols = n_u > n_v ? n_u : n_v;
    uint64_t wrk_size = m * max_cols;
    f64 *wrk1 = xcalloc(wrk_size, sizeof(f64));
    f64 *wrk2 = xcalloc(wrk_size, sizeof(f64));
    f64 *wrk3 = xcalloc(wrk_size, sizeof(f64));

    /* Fill V and U with random values */
    for (uint64_t i = 0; i < m * n_v; i++) V[i] = (f64)rand()/RAND_MAX - 0.5;
    for (uint64_t i = 0; i < m * n_u; i++) U[i] = (f64)rand()/RAND_MAX - 0.5;

    /* B-orthonormalize V using SVQB */
    d_svqb(m, n_v, 1e-14, 'n', V, wrk1, wrk2, wrk3, B);

    /* Compute signature matrix for V */
    f64 *sig = xcalloc(n_v * n_v, sizeof(f64));
    for (uint64_t j = 0; j < n_v; j++)
        B->matvec(B, &V[j * m], &wrk1[j * m]);
    d_gemm_tn(n_v, n_v, m, 1.0, V, wrk1, 0.0, sig);
    for (uint64_t j = 0; j < n_v; j++)
        for (uint64_t i = j + 1; i < n_v; i++)
            sig[i + j*n_v] = sig[j + i*n_v];

    f64 cross_pre = B_cross_error_d(m, n_u, n_v, U, V, B, wrk2);
    f64 norm_pre  = B_norm_error_d(m, n_u, U, B, wrk2);

    /* Call ortho_indefinite */
    d_ortho_indefinite(m, n_u, n_v, TOL_F64, 1e-14, U, V, sig, wrk1, wrk2, wrk3, B);

    f64 ortho_err = B_cross_error_d(m, n_u, n_v, U, V, B, wrk1);
    f64 norm_err  = B_norm_error_d(m, n_u, U, B, wrk1);
    printf("pre: cross=%.2e norm=%.2e  post: cross=%.2e norm=%.2e ", cross_pre, norm_pre, ortho_err, norm_err);
    ASSERT(ortho_err < TOL_F64 * n_u);
    ASSERT(norm_err < TOL_F64 * n_u);

    /* Cleanup */
    safe_free((void**)&V); safe_free((void**)&U); safe_free((void**)&sig);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3);
    linop_destroy_d(&B);
}

TEST(d_ortho_indefinite_no_sig) {
    /*
     * Test with sig=NULL (signature computed internally)
     */
    const uint64_t m = 80;
    const uint64_t n_v = 4;
    const uint64_t n_u = 6;
    const uint64_t n_pos = 50;

    LinearOperator_d_t *B = create_indef_B_d(m, n_pos);

    f64 *V = xcalloc(m * n_v, sizeof(f64));
    f64 *U = xcalloc(m * n_u, sizeof(f64));
    uint64_t max_cols = n_u > n_v ? n_u : n_v;
    uint64_t wrk_size = m * max_cols;
    f64 *wrk1 = xcalloc(wrk_size, sizeof(f64));
    f64 *wrk2 = xcalloc(wrk_size, sizeof(f64));
    f64 *wrk3 = xcalloc(wrk_size, sizeof(f64));

    for (uint64_t i = 0; i < m * n_v; i++) V[i] = (f64)rand()/RAND_MAX - 0.5;
    for (uint64_t i = 0; i < m * n_u; i++) U[i] = (f64)rand()/RAND_MAX - 0.5;

    /* B-orthonormalize V */
    d_svqb(m, n_v, 1e-14, 'n', V, wrk1, wrk2, wrk3, B);

    f64 cross_pre = B_cross_error_d(m, n_u, n_v, U, V, B, wrk2);
    f64 norm_pre  = B_norm_error_d(m, n_u, U, B, wrk2);

    /* Call ortho_indefinite with sig=NULL */
    d_ortho_indefinite(m, n_u, n_v, TOL_F64, 1e-14, U, V, NULL, wrk1, wrk2, wrk3, B);

    f64 ortho_err = B_cross_error_d(m, n_u, n_v, U, V, B, wrk1);
    f64 norm_err  = B_norm_error_d(m, n_u, U, B, wrk1);
    printf("pre: cross=%.2e norm=%.2e  post: cross=%.2e norm=%.2e ", cross_pre, norm_pre, ortho_err, norm_err);
    ASSERT(ortho_err < TOL_F64 * n_u);
    ASSERT(norm_err < TOL_F64 * n_u);

    safe_free((void**)&V); safe_free((void**)&U);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3);
    linop_destroy_d(&B);
}

/* ====================================================================
 * Double complex tests
 * ==================================================================== */

TEST(z_ortho_indefinite_basic) {
    const uint64_t m = 100;
    const uint64_t n_v = 5;
    const uint64_t n_u = 8;
    const uint64_t n_pos = 60;

    LinearOperator_z_t *B = create_indef_B_z(m, n_pos);

    c64 *V = xcalloc(m * n_v, sizeof(c64));
    c64 *U = xcalloc(m * n_u, sizeof(c64));
    uint64_t max_cols = n_u > n_v ? n_u : n_v;
    uint64_t wrk_size = m * max_cols;
    c64 *wrk1 = xcalloc(wrk_size, sizeof(c64));
    c64 *wrk2 = xcalloc(wrk_size, sizeof(c64));
    c64 *wrk3 = xcalloc(wrk_size, sizeof(c64));

    for (uint64_t i = 0; i < m * n_v; i++)
        V[i] = ((f64)rand()/RAND_MAX - 0.5) + I*((f64)rand()/RAND_MAX - 0.5);
    for (uint64_t i = 0; i < m * n_u; i++)
        U[i] = ((f64)rand()/RAND_MAX - 0.5) + I*((f64)rand()/RAND_MAX - 0.5);

    /* B-orthonormalize V */
    z_svqb(m, n_v, 1e-14, 'n', V, wrk1, wrk2, wrk3, B);

    f64 cross_pre = B_cross_error_z(m, n_u, n_v, U, V, B, wrk2);
    f64 norm_pre  = B_norm_error_z(m, n_u, U, B, wrk2);

    /* Call ortho_indefinite with sig=NULL */
    z_ortho_indefinite(m, n_u, n_v, TOL_F64, 1e-14, U, V, NULL, wrk1, wrk2, wrk3, B);

    f64 ortho_err = B_cross_error_z(m, n_u, n_v, U, V, B, wrk1);
    f64 norm_err  = B_norm_error_z(m, n_u, U, B, wrk1);
    printf("pre: cross=%.2e norm=%.2e  post: cross=%.2e norm=%.2e ", cross_pre, norm_pre, ortho_err, norm_err);
    ASSERT(ortho_err < TOL_F64 * n_u);
    ASSERT(norm_err < TOL_F64 * n_u);

    safe_free((void**)&V); safe_free((void**)&U);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3);
    linop_destroy_z(&B);
}

TEST(z_ortho_indefinite_larger) {
    const uint64_t m = 500;
    const uint64_t n_v = 10;
    const uint64_t n_u = 15;
    const uint64_t n_pos = 300;

    LinearOperator_z_t *B = create_indef_B_z(m, n_pos);

    c64 *V = xcalloc(m * n_v, sizeof(c64));
    c64 *U = xcalloc(m * n_u, sizeof(c64));
    uint64_t max_cols = n_u > n_v ? n_u : n_v;
    uint64_t wrk_size = m * max_cols;
    c64 *wrk1 = xcalloc(wrk_size, sizeof(c64));
    c64 *wrk2 = xcalloc(wrk_size, sizeof(c64));
    c64 *wrk3 = xcalloc(wrk_size, sizeof(c64));

    for (uint64_t i = 0; i < m * n_v; i++)
        V[i] = ((f64)rand()/RAND_MAX - 0.5) + I*((f64)rand()/RAND_MAX - 0.5);
    for (uint64_t i = 0; i < m * n_u; i++)
        U[i] = ((f64)rand()/RAND_MAX - 0.5) + I*((f64)rand()/RAND_MAX - 0.5);

    z_svqb(m, n_v, 1e-14, 'n', V, wrk1, wrk2, wrk3, B);

    f64 cross_pre = B_cross_error_z(m, n_u, n_v, U, V, B, wrk2);
    f64 norm_pre  = B_norm_error_z(m, n_u, U, B, wrk2);

    z_ortho_indefinite(m, n_u, n_v, TOL_F64, 1e-14, U, V, NULL, wrk1, wrk2, wrk3, B);

    f64 ortho_err = B_cross_error_z(m, n_u, n_v, U, V, B, wrk1);
    f64 norm_err  = B_norm_error_z(m, n_u, U, B, wrk1);
    printf("pre: cross=%.2e norm=%.2e  post: cross=%.2e norm=%.2e ", cross_pre, norm_pre, ortho_err, norm_err);
    ASSERT(ortho_err < TOL_F64 * n_u);
    ASSERT(norm_err < TOL_F64 * n_u);

    safe_free((void**)&V); safe_free((void**)&U);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3);
    linop_destroy_z(&B);
}

/* ====================================================================
 * B=NULL tests (standard orthogonalization path)
 * ==================================================================== */

TEST(d_ortho_indefinite_no_B) {
    const uint64_t m = 80, n_u = 6, n_v = 4;
    const uint64_t max_n = n_u > n_v ? n_u : n_v;
    const uint64_t wrk_size = m * max_n;

    f64 *V    = xcalloc(m * n_v, sizeof(f64));
    f64 *U    = xcalloc(m * n_u, sizeof(f64));
    f64 *wrk1 = xcalloc(wrk_size, sizeof(f64));
    f64 *wrk2 = xcalloc(wrk_size, sizeof(f64));
    f64 *wrk3 = xcalloc(wrk_size, sizeof(f64));

    for (uint64_t i = 0; i < m * n_v; i++) V[i] = (f64)rand()/RAND_MAX - 0.5;
    for (uint64_t i = 0; i < m * n_u; i++) U[i] = (f64)rand()/RAND_MAX - 0.5;

    /* Standard orthonormalize V (B=NULL) */
    d_svqb(m, n_v, 1e-14, 'n', V, wrk1, wrk2, wrk3, NULL);

    f64 cross_pre = cross_error_d(m, n_u, n_v, U, V, wrk2);
    f64 norm_pre  = norm_error_d(m, n_u, U, wrk2);

    /* ortho_indefinite with B=NULL, sig=NULL reduces to standard orth */
    d_ortho_indefinite(m, n_u, n_v, TOL_F64, 1e-14, U, V, NULL, wrk1, wrk2, wrk3, NULL);

    f64 cross = cross_error_d(m, n_u, n_v, U, V, wrk2);
    f64 norm  = norm_error_d(m, n_u, U, wrk2);
    printf("pre: cross=%.2e norm=%.2e  post: cross=%.2e norm=%.2e ", cross_pre, norm_pre, cross, norm);
    ASSERT(cross < TOL_F64 * n_u);
    ASSERT(norm < TOL_F64 * n_u);

    safe_free((void**)&V); safe_free((void**)&U);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3);
}

TEST(z_ortho_indefinite_no_B) {
    const uint64_t m = 80, n_u = 6, n_v = 4;
    const uint64_t max_n = n_u > n_v ? n_u : n_v;
    const uint64_t wrk_size = m * max_n;

    c64 *V    = xcalloc(m * n_v, sizeof(c64));
    c64 *U    = xcalloc(m * n_u, sizeof(c64));
    c64 *wrk1 = xcalloc(wrk_size, sizeof(c64));
    c64 *wrk2 = xcalloc(wrk_size, sizeof(c64));
    c64 *wrk3 = xcalloc(wrk_size, sizeof(c64));

    for (uint64_t i = 0; i < m * n_v; i++)
        V[i] = ((f64)rand()/RAND_MAX - 0.5) + I*((f64)rand()/RAND_MAX - 0.5);
    for (uint64_t i = 0; i < m * n_u; i++)
        U[i] = ((f64)rand()/RAND_MAX - 0.5) + I*((f64)rand()/RAND_MAX - 0.5);

    z_svqb(m, n_v, 1e-14, 'n', V, wrk1, wrk2, wrk3, NULL);

    f64 cross_pre = cross_error_z(m, n_u, n_v, U, V, wrk2);
    f64 norm_pre  = norm_error_z(m, n_u, U, wrk2);

    z_ortho_indefinite(m, n_u, n_v, TOL_F64, 1e-14, U, V, NULL, wrk1, wrk2, wrk3, NULL);

    f64 cross = cross_error_z(m, n_u, n_v, U, V, wrk2);
    f64 norm  = norm_error_z(m, n_u, U, wrk2);
    printf("pre: cross=%.2e norm=%.2e  post: cross=%.2e norm=%.2e ", cross_pre, norm_pre, cross, norm);
    ASSERT(cross < TOL_F64 * n_u);
    ASSERT(norm < TOL_F64 * n_u);

    safe_free((void**)&V); safe_free((void**)&U);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3);
}

/* ====================================================================
 * Main
 * ==================================================================== */

int main(void) {
    srand((unsigned)time(NULL));

    printf("Indefinite orthogonalization (double) tests:\n");
    RUN(d_ortho_indefinite_basic);
    RUN(d_ortho_indefinite_no_sig);
    RUN(d_ortho_indefinite_no_B);

    printf("\nIndefinite orthogonalization (double complex) tests:\n");
    RUN(z_ortho_indefinite_basic);
    RUN(z_ortho_indefinite_larger);
    RUN(z_ortho_indefinite_no_B);

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}
