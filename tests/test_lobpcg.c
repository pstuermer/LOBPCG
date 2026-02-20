/**
 * @file test_lobpcg.c
 * @brief Integration tests for main LOBPCG solver
 *
 * Uses dense 4x4 and 6x6 reference matrices (same as test_rayleigh_ritz.c)
 * plus a 1D Laplacian operator. Verifies eigenvalue accuracy, orthonormality,
 * and Rayleigh quotient diagonality.
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

#define PI 3.14159265358979323846

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

#define ASSERT_NEAR(a, b, tol) ASSERT(fabs((a) - (b)) < (tol))

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
 * Laplacian matvec: -d^2/dx^2 with Dirichlet BC, tridiagonal [-1,2,-1]/h^2
 * ================================================================ */

typedef struct {
    uint64_t n;
    f64 h2_inv;
} laplacian_ctx_t;

static void laplacian_matvec_d(const LinearOperator_d_t *op,
                               const f64 *x, f64 *y) {
    laplacian_ctx_t *ctx = (laplacian_ctx_t *)op->ctx;
    uint64_t n = ctx->n;
    f64 c = ctx->h2_inv;
    y[0] = c * (2.0 * x[0] - x[1]);
    for (uint64_t i = 1; i < n - 1; i++)
        y[i] = c * (-x[i-1] + 2.0 * x[i] - x[i+1]);
    y[n-1] = c * (-x[n-2] + 2.0 * x[n-1]);
}

/* ================================================================
 * Reference matrices (column-major, same as test_rayleigh_ritz.c)
 * ================================================================ */

static const f64 A4x4[16] = {
    4.0, 1.0, 2.0, 0.0,
    1.0, 3.0, 0.0, 1.0,
    2.0, 0.0, 5.0, 2.0,
    0.0, 1.0, 2.0, 6.0
};

static const f64 A6x6[36] = {
    4.0, 1.0, 2.0, 0.0, 1.0, 0.5,
    1.0, 3.0, 0.0, 1.0, 0.5, 0.0,
    2.0, 0.0, 5.0, 2.0, 1.0, 1.0,
    0.0, 1.0, 2.0, 6.0, 1.5, 0.0,
    1.0, 0.5, 1.0, 1.5, 5.0, 2.0,
    0.5, 0.0, 1.0, 0.0, 2.0, 4.0
};

/* Exact eigenvalues from dsyev */
static const f64 eigvals_4x4[4] = {
    1.338399579631295e+00, 3.463077212970466e+00,
    5.000000000000000e+00, 8.198523207398235e+00
};

static const f64 eigvals_6x6[6] = {
    1.208742643127633e+00, 2.230197331224639e+00,
    3.615464945758393e+00, 4.717703764957660e+00,
    5.517221003524097e+00, 9.710670311407574e+00
};

/* ================================================================
 * Verification helpers
 * ================================================================ */

/* ||X^T*A*X - diag(eigVal)||_F */
static f64 rayleigh_diag_d(uint64_t n, uint64_t nev, const f64 *A_mat,
                           const f64 *X, const f64 *eigVal) {
    f64 *AX = xcalloc(n * nev, sizeof(f64));
    f64 *G  = xcalloc(nev * nev, sizeof(f64));
    d_gemm_nn(n, nev, n, 1.0, A_mat, X, 0.0, AX);
    d_gemm_tn(nev, nev, n, 1.0, X, AX, 0.0, G);
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

/* ||X^T*X - I||_F */
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
 * Test 1: d_lobpcg on 4x4 dense matrix, nev=1
 * 3*sizeSub <= n: 3*1=3 <= 4
 * ================================================================ */
TEST(d_lobpcg_4x4) {
    const uint64_t n = 4, nev = 1, sizeSub = 1;

    dense_ctx_d_t *ctx = xcalloc(1, sizeof(dense_ctx_d_t));
    ctx->n = n;
    ctx->A = xcalloc(n * n, sizeof(f64));
    memcpy(ctx->A, A4x4, n * n * sizeof(f64));

    LinearOperator_d_t A = { .rows = n, .cols = n, .matvec = dense_matvec_d,
                             .ctx = (linop_ctx_t *)ctx };

    d_lobpcg_t *alg = d_lobpcg_alloc(n, nev, sizeSub);
    alg->A = &A;
    alg->B = NULL;
    alg->T = NULL;
    alg->maxIter = 100;
    alg->tol = 1e-5;

    d_lobpcg(alg);

    printf("conv=%lu ", (unsigned long)alg->converged);
    ASSERT(alg->converged == nev);

    ASSERT_NEAR(alg->eigVals[0], eigvals_4x4[0], 1e-8);

    f64 *X = alg->S;
    f64 orth = ortho_self_d(n, nev, X);
    printf("orth=%.2e ", orth);
    ASSERT(orth < 1e-8);

    f64 rq = rayleigh_diag_d(n, nev, ctx->A, X, alg->eigVals);
    printf("rq=%.2e ", rq);
    ASSERT(rq < 1e-8);

    d_lobpcg_free(&alg);
    safe_free((void**)&ctx->A); safe_free((void**)&ctx);
}

/* ================================================================
 * Test 2: z_lobpcg on 4x4 dense matrix (complex), nev=1
 * 3*sizeSub <= n: 3*1=3 <= 4
 * ================================================================ */
TEST(z_lobpcg_4x4) {
    const uint64_t n = 4, nev = 1, sizeSub = 1;

    dense_ctx_z_t *ctx = xcalloc(1, sizeof(dense_ctx_z_t));
    ctx->n = n;
    ctx->A = xcalloc(n * n, sizeof(c64));
    for (uint64_t i = 0; i < n * n; i++) ctx->A[i] = A4x4[i] + 0*I;

    LinearOperator_z_t A = { .rows = n, .cols = n, .matvec = dense_matvec_z,
                             .ctx = (linop_ctx_t *)ctx };

    z_lobpcg_t *alg = z_lobpcg_alloc(n, nev, sizeSub);
    alg->A = &A;
    alg->B = NULL;
    alg->T = NULL;
    alg->maxIter = 100;
    alg->tol = 1e-5;

    z_lobpcg(alg);

    printf("conv=%lu ", (unsigned long)alg->converged);
    ASSERT(alg->converged == nev);

    ASSERT_NEAR(alg->eigVals[0], eigvals_4x4[0], 1e-4);

    c64 *X = alg->S;
    f64 orth = ortho_self_z(n, nev, X);
    printf("orth=%.2e ", orth);
    ASSERT(orth < 1e-8);

    f64 rq = rayleigh_diag_z(n, nev, ctx->A, X, alg->eigVals);
    printf("rq=%.2e ", rq);
    ASSERT(rq < 1e-8);

    z_lobpcg_free(&alg);
    safe_free((void**)&ctx->A); safe_free((void**)&ctx);
}

/* ================================================================
 * Test 3: d_lobpcg on 6x6 dense matrix, nev=1, sizeSub=2
 * 3*sizeSub <= n: 3*2=6 <= 6
 * ================================================================ */
TEST(d_lobpcg_6x6) {
    const uint64_t n = 6, nev = 1, sizeSub = 2;

    dense_ctx_d_t *ctx = xcalloc(1, sizeof(dense_ctx_d_t));
    ctx->n = n;
    ctx->A = xcalloc(n * n, sizeof(f64));
    memcpy(ctx->A, A6x6, n * n * sizeof(f64));

    LinearOperator_d_t A = { .rows = n, .cols = n, .matvec = dense_matvec_d,
                             .ctx = (linop_ctx_t *)ctx };

    d_lobpcg_t *alg = d_lobpcg_alloc(n, nev, sizeSub);
    alg->A = &A;
    alg->B = NULL;
    alg->T = NULL;
    alg->maxIter = 100;
    alg->tol = 1e-5;

    d_lobpcg(alg);

    printf("conv=%lu ev0=%.6e ", (unsigned long)alg->converged,
           alg->eigVals[0]);
    ASSERT(alg->converged == nev);

    ASSERT_NEAR(alg->eigVals[0], eigvals_6x6[0], 1e-6);

    f64 *X = alg->S;
    f64 orth = ortho_self_d(n, nev, X);
    printf("orth=%.2e ", orth);
    ASSERT(orth < 1e-8);

    f64 rq = rayleigh_diag_d(n, nev, ctx->A, X, alg->eigVals);
    printf("rq=%.2e ", rq);
    ASSERT(rq < 1e-8);

    d_lobpcg_free(&alg);
    safe_free((void**)&ctx->A); safe_free((void**)&ctx);
}

/* ================================================================
 * Test 4: d_lobpcg on 6x6 dense matrix, nev=2, sizeSub=2
 * 3*sizeSub <= n: 3*2=6 <= 6
 * ================================================================ */
TEST(d_lobpcg_6x6_nev2) {
    const uint64_t n = 6, nev = 2, sizeSub = 2;

    dense_ctx_d_t *ctx = xcalloc(1, sizeof(dense_ctx_d_t));
    ctx->n = n;
    ctx->A = xcalloc(n * n, sizeof(f64));
    memcpy(ctx->A, A6x6, n * n * sizeof(f64));

    LinearOperator_d_t A = { .rows = n, .cols = n, .matvec = dense_matvec_d,
                             .ctx = (linop_ctx_t *)ctx };

    d_lobpcg_t *alg = d_lobpcg_alloc(n, nev, sizeSub);
    alg->A = &A;
    alg->B = NULL;
    alg->T = NULL;
    alg->maxIter = 100;
    alg->tol = 1e-5;

    d_lobpcg(alg);

    printf("conv=%lu ", (unsigned long)alg->converged);
    ASSERT(alg->converged == nev);

    ASSERT_NEAR(alg->eigVals[0], eigvals_6x6[0], 1e-6);
    ASSERT_NEAR(alg->eigVals[1], eigvals_6x6[1], 1e-6);

    f64 *X = alg->S;
    f64 orth = ortho_self_d(n, nev, X);
    printf("orth=%.2e ", orth);
    ASSERT(orth < 1e-8);

    f64 rq = rayleigh_diag_d(n, nev, ctx->A, X, alg->eigVals);
    printf("rq=%.2e ", rq);
    ASSERT(rq < 1e-8);

    d_lobpcg_free(&alg);
    safe_free((void**)&ctx->A); safe_free((void**)&ctx);
}

/* ================================================================
 * Test 5: d_lobpcg on 1D Laplacian, n=100, nev=5
 * ================================================================ */
TEST(d_lobpcg_laplacian) {
    const uint64_t n = 100, nev = 3;

    f64 L = 1.0;
    f64 h = L / (n + 1);
    laplacian_ctx_t *ctx = xcalloc(1, sizeof(laplacian_ctx_t));
    ctx->n = n;
    ctx->h2_inv = 1.0 / (h * h);

    LinearOperator_d_t A = { .rows = n, .cols = n,
                             .matvec = (void (*)(const LinearOperator_d_t *, f64 *, f64 *))laplacian_matvec_d,
                             .ctx = (linop_ctx_t *)ctx };

    const uint64_t sizeSub = 5;
    d_lobpcg_t *alg = d_lobpcg_alloc(n, nev, sizeSub);
    alg->A = &A;
    alg->B = NULL;
    alg->T = NULL;
    alg->maxIter = 500;
    alg->tol = 1e-4;

    d_lobpcg(alg);

    printf("conv=%lu iter=%lu ", (unsigned long)alg->converged,
           (unsigned long)alg->iter);
    ASSERT(alg->converged == nev);

    /* Check against analytical eigenvalues: lambda_k = (k*pi)^2 for 1D Laplacian
     * (discretization introduces small error, so use 1% relative tolerance) */
    for (uint64_t k = 1; k <= nev; k++) {
        f64 exact = (k * PI) * (k * PI);
        f64 rel_err = fabs(alg->eigVals[k-1] - exact) / exact;
        ASSERT(rel_err < 0.01);
    }

    d_lobpcg_free(&alg);
    safe_free((void**)&ctx);
}

/* ================================================================ */
int main(void) {
    printf("LOBPCG dense tests:\n");
    RUN(d_lobpcg_4x4);
    RUN(z_lobpcg_4x4);
    RUN(d_lobpcg_6x6);
    RUN(d_lobpcg_6x6_nev2);

    printf("\nLOBPCG Laplacian test:\n");
    RUN(d_lobpcg_laplacian);

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}
