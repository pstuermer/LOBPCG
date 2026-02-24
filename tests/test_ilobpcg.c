/**
 * @file test_ilobpcg.c
 * @brief Integration tests for indefinite LOBPCG solver
 *
 * Test 1: Diagonal A + indefinite diagonal B
 *   A = diag(1,2,...,n), B = diag(+1,...,+1,-1,...,-1)
 *   Expected smallest positive eigenvalues: 1, 2, 3
 *
 * Test 2: Block-Laplacian A + block-permutation B
 *   A = {{K,0},{0,K}}, B = {{0,I},{I,0}} where K = 1D Laplacian
 *   Eigenvalues are ±μ_k where μ_k = (kπ)^2
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
 * Diagonal operators for test 1
 * ================================================================ */

typedef struct {
    uint64_t n;
} diag_ctx_t;

static void diag_A_matvec_d(const LinearOperator_d_t *op,
                             f64 *restrict x, f64 *restrict y) {
    diag_ctx_t *ctx = (diag_ctx_t *)op->ctx;
    for (uint64_t i = 0; i < ctx->n; i++)
        y[i] = (f64)(i + 1) * x[i];
}

static void diag_B_matvec_d(const LinearOperator_d_t *op,
                             f64 *restrict x, f64 *restrict y) {
    diag_ctx_t *ctx = (diag_ctx_t *)op->ctx;
    uint64_t half = ctx->n / 2;
    for (uint64_t i = 0; i < ctx->n; i++)
        y[i] = (i < half) ? x[i] : -x[i];
}

/* ================================================================
 * Block-Laplacian A: A = {{K,0},{0,K}}, K = tridiag[-1,2,-1]/h^2
 * ================================================================ */

typedef struct {
    uint64_t m;
    f64 h2_inv;
} block_laplacian_ctx_t;

static void block_laplacian_matvec_d(const LinearOperator_d_t *op,
                                     f64 *restrict x, f64 *restrict y) {
    block_laplacian_ctx_t *ctx = (block_laplacian_ctx_t *)op->ctx;
    uint64_t m = ctx->m;
    f64 c = ctx->h2_inv;

    /* Apply K to top half x[0:m] → y[0:m] and bottom half x[m:2m] → y[m:2m] */
    for (uint64_t blk = 0; blk < 2; blk++) {
        uint64_t off = blk * m;
        y[off] = c * (2.0 * x[off] - x[off + 1]);
        for (uint64_t i = 1; i < m - 1; i++)
            y[off + i] = c * (-x[off + i - 1] + 2.0 * x[off + i] - x[off + i + 1]);
        y[off + m - 1] = c * (-x[off + m - 2] + 2.0 * x[off + m - 1]);
    }
}

/* ================================================================
 * Block-permutation B: B = {{0,I},{I,0}}, swaps top/bottom halves
 * ================================================================ */

typedef struct {
    uint64_t m;
} block_perm_ctx_t;

static void block_perm_matvec_d(const LinearOperator_d_t *op,
                                f64 *restrict x, f64 *restrict y) {
    block_perm_ctx_t *ctx = (block_perm_ctx_t *)op->ctx;
    uint64_t m = ctx->m;
    memcpy(y,     x + m, m * sizeof(f64));  /* y[0:m]   = x[m:2m] */
    memcpy(y + m, x,     m * sizeof(f64));  /* y[m:2m]  = x[0:m]  */
}

/* ================================================================
 * Test 1: Diagonal indefinite eigenvalue problem
 * ================================================================ */
TEST(d_ilobpcg_diagonal) {
    const uint64_t n = 100, nev = 3, sizeSub = 3;
    const uint64_t maxIter = 200;
    const f64 tol = 1e-6;

    diag_ctx_t *ctx_a = xcalloc(1, sizeof(diag_ctx_t));
    ctx_a->n = n;
    LinearOperator_d_t A = { .rows = n, .cols = n, .matvec = diag_A_matvec_d,
                             .ctx = (linop_ctx_t *)ctx_a };

    diag_ctx_t *ctx_b = xcalloc(1, sizeof(diag_ctx_t));
    ctx_b->n = n;
    LinearOperator_d_t B = { .rows = n, .cols = n, .matvec = diag_B_matvec_d,
                             .ctx = (linop_ctx_t *)ctx_b };

    d_lobpcg_t *alg = d_ilobpcg_alloc(n, nev, sizeSub);
    alg->A = &A;
    alg->B = &B;
    alg->T = NULL;
    alg->maxIter = maxIter;
    alg->tol = tol;

    d_ilobpcg(alg);

    printf("conv=%lu iter=%lu ", (unsigned long)alg->converged,
           (unsigned long)alg->iter);
    ASSERT(alg->converged == nev);

    /* Expected: smallest positive eigenvalues are 1, 2, 3 */
    for (uint64_t k = 0; k < nev; k++) {
        f64 exact = (f64)(k + 1);
        f64 rel_err = fabs(alg->eigVals[k] - exact) / exact;
        ASSERT(rel_err < 0.01);
    }

    d_lobpcg_free(&alg);
    safe_free((void **)&ctx_a);
    safe_free((void **)&ctx_b);
}

/* ================================================================
 * Test 2: Block-Laplacian + block-permutation B
 *   A = {{K,0},{0,K}}, B = {{0,I},{I,0}}
 *   Eigenvalues: ±μ_k where μ_k = (kπ/(m+1))^2 * (m+1)^2 = (kπ)^2
 *   Smallest positive: (π)^2, (2π)^2, (3π)^2
 * ================================================================ */
TEST(d_ilobpcg_block_laplacian) {
    const uint64_t m = 50;
    const uint64_t n = 2 * m;
    const uint64_t nev = 3, sizeSub = 5;
    const uint64_t maxIter = 500;
    const f64 tol = 1e-4;

    f64 L = 1.0;
    f64 h = L / (m + 1);

    /* Block-Laplacian A */
    block_laplacian_ctx_t *ctx_a = xcalloc(1, sizeof(block_laplacian_ctx_t));
    ctx_a->m = m;
    ctx_a->h2_inv = 1.0 / (h * h);
    LinearOperator_d_t A = { .rows = n, .cols = n,
                             .matvec = (matvec_func_d_t)block_laplacian_matvec_d,
                             .ctx = (linop_ctx_t *)ctx_a };

    /* Block-permutation B */
    block_perm_ctx_t *ctx_b = xcalloc(1, sizeof(block_perm_ctx_t));
    ctx_b->m = m;
    LinearOperator_d_t B = { .rows = n, .cols = n,
                             .matvec = (matvec_func_d_t)block_perm_matvec_d,
                             .ctx = (linop_ctx_t *)ctx_b };

    d_lobpcg_t *alg = d_ilobpcg_alloc(n, nev, sizeSub);
    alg->A = &A;
    alg->B = &B;
    alg->T = NULL;
    alg->maxIter = maxIter;
    alg->tol = tol;

    d_ilobpcg(alg);

    printf("conv=%lu iter=%lu ", (unsigned long)alg->converged,
           (unsigned long)alg->iter);
    ASSERT(alg->converged == nev);

    /* Check eigenvalues ≈ (kπ)^2, 1% relative tolerance
     * (discretization error ~ O(h^2) is small for m=50) */
    for (uint64_t k = 1; k <= nev; k++) {
        f64 exact = (k * PI) * (k * PI);
        f64 rel_err = fabs(alg->eigVals[k - 1] - exact) / exact;
        ASSERT(rel_err < 0.01);
    }

    d_lobpcg_free(&alg);
    safe_free((void **)&ctx_a);
    safe_free((void **)&ctx_b);
}

/* ================================================================ */
int main(void) {
    printf("Indefinite LOBPCG tests:\n");
    RUN(d_ilobpcg_diagonal);
    RUN(d_ilobpcg_block_laplacian);

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}
