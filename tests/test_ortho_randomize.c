/**
 * @file test_ortho_randomize.c
 * @brief Test ortho_randomize B-orthogonalization against V
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <complex.h>
#include "types.h"
#include "lobpcg.h"
#include "lobpcg/linop.h"
#include "lobpcg/blas_wrapper.h"
#include "lobpcg/memory.h"

#define TOL 1e-12

static int tests_passed = 0, tests_failed = 0;

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
 * PD diagonal B operator: B = 2*I
 * ==================================================================== */

typedef struct { uint64_t n; } pd_diag_ctx_t;

static void pd_diag_matvec_d(const LinearOperator_d_t *op, f64 *x, f64 *y) {
    pd_diag_ctx_t *ctx = (pd_diag_ctx_t *)op->ctx->data;
    for (uint64_t i = 0; i < ctx->n; i++) y[i] = 2.0 * x[i];
}

static void pd_diag_matvec_z(const LinearOperator_z_t *op, c64 *x, c64 *y) {
    pd_diag_ctx_t *ctx = (pd_diag_ctx_t *)op->ctx->data;
    for (uint64_t i = 0; i < ctx->n; i++) y[i] = 2.0 * x[i];
}

static void pd_diag_cleanup(linop_ctx_t *ctx) {
    if (ctx && ctx->data) safe_free((void**)&ctx->data);
    if (ctx) safe_free((void**)&ctx);
}

static LinearOperator_d_t *create_pd_B_d(uint64_t n) {
    linop_ctx_t *ctx = xcalloc(1, sizeof(linop_ctx_t));
    pd_diag_ctx_t *data = xcalloc(1, sizeof(pd_diag_ctx_t));
    data->n = n; ctx->data = data; ctx->data_size = sizeof(pd_diag_ctx_t);
    return linop_create_d(n, n, pd_diag_matvec_d, pd_diag_cleanup, ctx);
}

static LinearOperator_z_t *create_pd_B_z(uint64_t n) {
    linop_ctx_t *ctx = xcalloc(1, sizeof(linop_ctx_t));
    pd_diag_ctx_t *data = xcalloc(1, sizeof(pd_diag_ctx_t));
    data->n = n; ctx->data = data; ctx->data_size = sizeof(pd_diag_ctx_t);
    return linop_create_z(n, n, pd_diag_matvec_z, pd_diag_cleanup, ctx);
}

/* ====================================================================
 * Error helpers
 * ==================================================================== */

/* B=NULL: ||V^H*U||_F;  wrk: >= n_v*n_u elements */
static f64 cross_error_d(uint64_t m, uint64_t n_u, uint64_t n_v,
                          const f64 *U, const f64 *V, f64 *wrk) {
    d_gemm_tn(n_v, n_u, m, 1.0, V, U, 0.0, wrk);
    return d_nrm2(n_v * n_u, wrk);
}

/* B=NULL: ||U^H*U - I||_F;  wrk: >= n*n elements */
static f64 norm_error_d(uint64_t m, uint64_t n, const f64 *U, f64 *wrk) {
    d_gemm_tn(n, n, m, 1.0, U, U, 0.0, wrk);
    for (uint64_t i = 0; i < n; i++) wrk[i + i*n] -= 1.0;
    return d_nrm2(n * n, wrk);
}

/* B!=NULL: ||V^H*B*U||_F;  wrk: >= m*n_u elements */
static f64 B_cross_error_d(uint64_t m, uint64_t n_u, uint64_t n_v,
                             const f64 *U, const f64 *V,
                             LinearOperator_d_t *B, f64 *wrk) {
    for (uint64_t j = 0; j < n_u; j++)
        B->matvec(B, (f64*)&U[j*m], &wrk[j*m]);
    f64 *G = xcalloc(n_v * n_u, sizeof(f64));
    d_gemm_tn(n_v, n_u, m, 1.0, V, wrk, 0.0, G);
    f64 err = d_nrm2(n_v * n_u, G);
    safe_free((void**)&G);
    return err;
}

/* B!=NULL: max(||off-diag(G)||_F, max_i ||G_ii|-1|) where G=U^H*B*U;
 * wrk: >= m*n elements */
static f64 B_norm_error_d(uint64_t m, uint64_t n,
                           const f64 *U, LinearOperator_d_t *B, f64 *wrk) {
    for (uint64_t j = 0; j < n; j++)
        B->matvec(B, (f64*)&U[j*m], &wrk[j*m]);
    f64 *G = xcalloc(n * n, sizeof(f64));
    d_gemm_tn(n, n, m, 1.0, U, wrk, 0.0, G);
    f64 off_diag = 0, diag_err = 0;
    for (uint64_t j = 0; j < n; j++)
        for (uint64_t i = 0; i < n; i++)
            if (i != j) off_diag += G[i + j*n] * G[i + j*n];
    off_diag = sqrt(off_diag);
    for (uint64_t i = 0; i < n; i++) {
        f64 d = fabs(fabs(G[i + i*n]) - 1.0);
        if (d > diag_err) diag_err = d;
    }
    safe_free((void**)&G);
    return off_diag > diag_err ? off_diag : diag_err;
}

/* Complex variants */
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

static f64 B_cross_error_z(uint64_t m, uint64_t n_u, uint64_t n_v,
                             const c64 *U, const c64 *V,
                             LinearOperator_z_t *B, c64 *wrk) {
    for (uint64_t j = 0; j < n_u; j++)
        B->matvec(B, (c64*)&U[j*m], &wrk[j*m]);
    c64 *G = xcalloc(n_v * n_u, sizeof(c64));
    z_gemm_hn(n_v, n_u, m, 1.0, V, wrk, 0.0, G);
    f64 err = z_nrm2(n_v * n_u, G);
    safe_free((void**)&G);
    return err;
}

static f64 B_norm_error_z(uint64_t m, uint64_t n,
                           const c64 *U, LinearOperator_z_t *B, c64 *wrk) {
    for (uint64_t j = 0; j < n; j++)
        B->matvec(B, (c64*)&U[j*m], &wrk[j*m]);
    c64 *G = xcalloc(n * n, sizeof(c64));
    z_gemm_hn(n, n, m, 1.0, U, wrk, 0.0, G);
    f64 off_diag = 0, diag_err = 0;
    for (uint64_t j = 0; j < n; j++)
        for (uint64_t i = 0; i < n; i++)
            if (i != j) off_diag += cabs(G[i + j*n]) * cabs(G[i + j*n]);
    off_diag = sqrt(off_diag);
    for (uint64_t i = 0; i < n; i++) {
        f64 d = fabs(cabs(G[i + i*n]) - 1.0);
        if (d > diag_err) diag_err = d;
    }
    safe_free((void**)&G);
    return off_diag > diag_err ? off_diag : diag_err;
}

/* ====================================================================
 * Tests
 * ==================================================================== */

TEST(d_ortho_randomize_no_B) {
    const uint64_t m = 100, n_u = 10, n_v = 10;
    const f64 eps = 1e-14;
    const uint64_t max_n = n_u > n_v ? n_u : n_v;

    f64 *U    = xcalloc(m * n_u, sizeof(f64));
    f64 *V    = xcalloc(m * n_v, sizeof(f64));
    f64 *wrk1 = xcalloc(m * (n_u + n_v), sizeof(f64));
    f64 *wrk2 = xcalloc(m * max_n, sizeof(f64));
    f64 *wrk3 = xcalloc(m * max_n, sizeof(f64));

    for (uint64_t i = 0; i < m * n_u; i++) U[i] = (f64)rand() / RAND_MAX;
    for (uint64_t i = 0; i < m * n_v; i++) V[i] = (f64)rand() / RAND_MAX;
    d_svqb(m, n_v, eps, 'n', V, wrk1, wrk2, wrk3, NULL);

    f64 cross_pre = cross_error_d(m, n_u, n_v, U, V, wrk2);
    f64 norm_pre  = norm_error_d(m, n_u, U, wrk2);

    d_ortho_randomize(m, n_u, n_v, eps, eps, U, V, wrk1, wrk2, wrk3, NULL);

    f64 cross = cross_error_d(m, n_u, n_v, U, V, wrk2);
    f64 norm  = norm_error_d(m, n_u, U, wrk2);
    printf("pre: cross=%.2e norm=%.2e  post: cross=%.2e norm=%.2e ", cross_pre, norm_pre, cross, norm);
    ASSERT(cross < TOL);
    ASSERT(norm < TOL);

    safe_free((void**)&U); safe_free((void**)&V);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3);
}

TEST(d_ortho_randomize_with_B) {
    const uint64_t m = 100, n_u = 10, n_v = 10;
    const f64 eps = 1e-14;
    const uint64_t max_n = n_u > n_v ? n_u : n_v;

    f64 *U    = xcalloc(m * n_u, sizeof(f64));
    f64 *V    = xcalloc(m * n_v, sizeof(f64));
    f64 *wrk1 = xcalloc(m * (n_u + n_v), sizeof(f64));
    f64 *wrk2 = xcalloc(m * max_n, sizeof(f64));
    f64 *wrk3 = xcalloc(m * max_n, sizeof(f64));
    LinearOperator_d_t *B = create_pd_B_d(m);

    for (uint64_t i = 0; i < m * n_u; i++) U[i] = (f64)rand() / RAND_MAX;
    for (uint64_t i = 0; i < m * n_v; i++) V[i] = (f64)rand() / RAND_MAX;
    d_svqb(m, n_v, eps, 'n', V, wrk1, wrk2, wrk3, B);

    f64 cross_pre = B_cross_error_d(m, n_u, n_v, U, V, B, wrk2);
    f64 norm_pre  = B_norm_error_d(m, n_u, U, B, wrk2);

    d_ortho_randomize(m, n_u, n_v, eps, eps, U, V, wrk1, wrk2, wrk3, B);

    f64 cross = B_cross_error_d(m, n_u, n_v, U, V, B, wrk2);
    f64 norm  = B_norm_error_d(m, n_u, U, B, wrk2);
    printf("pre: cross=%.2e norm=%.2e  post: cross=%.2e norm=%.2e ", cross_pre, norm_pre, cross, norm);
    ASSERT(cross < TOL);
    ASSERT(norm < TOL);

    safe_free((void**)&U); safe_free((void**)&V);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3);
    linop_destroy_d(&B);
}

TEST(z_ortho_randomize_no_B) {
    const uint64_t m = 100, n_u = 10, n_v = 10;
    const f64 eps = 1e-14;
    const uint64_t max_n = n_u > n_v ? n_u : n_v;

    c64 *U    = xcalloc(m * n_u, sizeof(c64));
    c64 *V    = xcalloc(m * n_v, sizeof(c64));
    c64 *wrk1 = xcalloc(m * (n_u + n_v), sizeof(c64));
    c64 *wrk2 = xcalloc(m * max_n, sizeof(c64));
    c64 *wrk3 = xcalloc(m * max_n, sizeof(c64));

    for (uint64_t i = 0; i < m * n_u; i++)
        U[i] = ((f64)rand()/RAND_MAX) + I*((f64)rand()/RAND_MAX);
    for (uint64_t i = 0; i < m * n_v; i++)
        V[i] = ((f64)rand()/RAND_MAX) + I*((f64)rand()/RAND_MAX);
    z_svqb(m, n_v, eps, 'n', V, wrk1, wrk2, wrk3, NULL);

    f64 cross_pre = cross_error_z(m, n_u, n_v, U, V, wrk2);
    f64 norm_pre  = norm_error_z(m, n_u, U, wrk2);

    z_ortho_randomize(m, n_u, n_v, eps, eps, U, V, wrk1, wrk2, wrk3, NULL);

    f64 cross = cross_error_z(m, n_u, n_v, U, V, wrk2);
    f64 norm  = norm_error_z(m, n_u, U, wrk2);
    printf("pre: cross=%.2e norm=%.2e  post: cross=%.2e norm=%.2e ", cross_pre, norm_pre, cross, norm);
    ASSERT(cross < TOL);
    ASSERT(norm < TOL);

    safe_free((void**)&U); safe_free((void**)&V);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3);
}

TEST(z_ortho_randomize_with_B) {
    const uint64_t m = 100, n_u = 10, n_v = 10;
    const f64 eps = 1e-14;
    const uint64_t max_n = n_u > n_v ? n_u : n_v;

    c64 *U    = xcalloc(m * n_u, sizeof(c64));
    c64 *V    = xcalloc(m * n_v, sizeof(c64));
    c64 *wrk1 = xcalloc(m * (n_u + n_v), sizeof(c64));
    c64 *wrk2 = xcalloc(m * max_n, sizeof(c64));
    c64 *wrk3 = xcalloc(m * max_n, sizeof(c64));
    LinearOperator_z_t *B = create_pd_B_z(m);

    for (uint64_t i = 0; i < m * n_u; i++)
        U[i] = ((f64)rand()/RAND_MAX) + I*((f64)rand()/RAND_MAX);
    for (uint64_t i = 0; i < m * n_v; i++)
        V[i] = ((f64)rand()/RAND_MAX) + I*((f64)rand()/RAND_MAX);
    z_svqb(m, n_v, eps, 'n', V, wrk1, wrk2, wrk3, B);

    f64 cross_pre = B_cross_error_z(m, n_u, n_v, U, V, B, wrk2);
    f64 norm_pre  = B_norm_error_z(m, n_u, U, B, wrk2);

    z_ortho_randomize(m, n_u, n_v, eps, eps, U, V, wrk1, wrk2, wrk3, B);

    f64 cross = B_cross_error_z(m, n_u, n_v, U, V, B, wrk2);
    f64 norm  = B_norm_error_z(m, n_u, U, B, wrk2);
    printf("pre: cross=%.2e norm=%.2e  post: cross=%.2e norm=%.2e ", cross_pre, norm_pre, cross, norm);
    ASSERT(cross < TOL);
    ASSERT(norm < TOL);

    safe_free((void**)&U); safe_free((void**)&V);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3);
    linop_destroy_z(&B);
}

int main(void) {
    srand((unsigned)time(NULL));

    printf("ortho_randomize tests:\n");
    RUN(d_ortho_randomize_no_B);
    RUN(d_ortho_randomize_with_B);
    RUN(z_ortho_randomize_no_B);
    RUN(z_ortho_randomize_with_B);

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n");
    return tests_failed > 0 ? 1 : 0;
}
