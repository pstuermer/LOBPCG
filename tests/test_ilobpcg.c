/**
 * @file test_ilobpcg.c
 * @brief Integration tests for indefinite LOBPCG solver
 *
 * Test 1 (d): Block-Laplacian A + block-permutation B (double)
 *   A = {{K,0},{0,K}}, B = {{0,I},{I,0}} where K = 1D Laplacian
 *   Eigenvalues are +/- mu_k where mu_k = (k*pi)^2
 *   B-positive init: X_k = [e_k; e_k] so v^T*B*v = 2 > 0
 *
 * Test 2 (z): Same problem with complex double types
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

#include "test_macros.h"

/* ================================================================
 * Block-Laplacian A (double): A = {{K,0},{0,K}}, K = tridiag[-1,2,-1]/h^2
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

    /* Apply K to top half x[0:m] -> y[0:m] and bottom half x[m:2m] -> y[m:2m] */
    for (uint64_t blk = 0; blk < 2; blk++) {
        uint64_t off = blk * m;
        y[off] = c * (2.0 * x[off] - x[off + 1]);
        for (uint64_t i = 1; i < m - 1; i++)
            y[off + i] = c * (-x[off + i - 1] + 2.0 * x[off + i] - x[off + i + 1]);
        y[off + m - 1] = c * (-x[off + m - 2] + 2.0 * x[off + m - 1]);
    }
}

/* ================================================================
 * Block-permutation B (double): B = {{0,I},{I,0}}, swaps top/bottom halves
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
 * Block-Laplacian A (complex double): same structure, c64 types
 * ================================================================ */

static void block_laplacian_matvec_z(const LinearOperator_z_t *op,
                                     c64 *restrict x, c64 *restrict y) {
    block_laplacian_ctx_t *ctx = (block_laplacian_ctx_t *)op->ctx;
    uint64_t m = ctx->m;
    f64 c = ctx->h2_inv;

    for (uint64_t blk = 0; blk < 2; blk++) {
        uint64_t off = blk * m;
        y[off] = c * (2.0 * x[off] - x[off + 1]);
        for (uint64_t i = 1; i < m - 1; i++)
            y[off + i] = c * (-x[off + i - 1] + 2.0 * x[off + i] - x[off + i + 1]);
        y[off + m - 1] = c * (-x[off + m - 2] + 2.0 * x[off + m - 1]);
    }
}

/* ================================================================
 * Block-permutation B (complex double): swaps top/bottom halves
 * ================================================================ */

static void block_perm_matvec_z(const LinearOperator_z_t *op,
                                c64 *restrict x, c64 *restrict y) {
    block_perm_ctx_t *ctx = (block_perm_ctx_t *)op->ctx;
    uint64_t m = ctx->m;
    memcpy(y,     x + m, m * sizeof(c64));
    memcpy(y + m, x,     m * sizeof(c64));
}

/* ================================================================
 * Test 1: Block-Laplacian + block-permutation B (double)
 *   A = {{K,0},{0,K}}, B = {{0,I},{I,0}}
 *   B-positive init: X_k = [e_k; e_k], v^T*B*v = 2 > 0
 *   Smallest positive eigenvalues: (pi)^2, (2*pi)^2, (3*pi)^2
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

    /* B-positive initialization: deterministic random vectors in V+.
     * For B={{0,I},{I,0}}, V+ = {[u;u]}: set top = bottom half.
     * v^T*B*v = 2*u^T*u > 0, and dense u overlaps all V+ eigenmodes. */
    srand(42);
    for (uint64_t k = 0; k < sizeSub; k++) {
        for (uint64_t j = 0; j < m; j++) {
            f64 val = (f64)rand() / RAND_MAX - 0.5;
            alg->S[j + k * n]     = val;
            alg->S[m + j + k * n] = val;
        }
    }

    d_ilobpcg(alg);

    printf("conv=%lu iter=%lu ", (unsigned long)alg->converged,
           (unsigned long)alg->iter);
    ASSERT(alg->converged == nev);

    /* Check eigenvalues are positive and match (k*pi)^2 */
    for (uint64_t k = 1; k <= nev; k++) {
        f64 exact = (k * PI) * (k * PI);
        ASSERT(alg->eigVals[k - 1] > 0);
        f64 rel_err = fabs(alg->eigVals[k - 1] - exact) / exact;
        ASSERT(rel_err < 0.01);
    }

    d_lobpcg_free(&alg);
    safe_free((void **)&ctx_a);
    safe_free((void **)&ctx_b);
}

/* ================================================================
 * Test 2: Block-Laplacian + block-permutation B (complex double)
 *   Same problem as Test 1 but with c64 types
 * ================================================================ */
TEST(z_ilobpcg_block_laplacian) {
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
    LinearOperator_z_t A = { .rows = n, .cols = n,
                             .matvec = (matvec_func_z_t)block_laplacian_matvec_z,
                             .ctx = (linop_ctx_t *)ctx_a };

    /* Block-permutation B */
    block_perm_ctx_t *ctx_b = xcalloc(1, sizeof(block_perm_ctx_t));
    ctx_b->m = m;
    LinearOperator_z_t B = { .rows = n, .cols = n,
                             .matvec = (matvec_func_z_t)block_perm_matvec_z,
                             .ctx = (linop_ctx_t *)ctx_b };

    z_lobpcg_t *alg = z_ilobpcg_alloc(n, nev, sizeSub);
    alg->A = &A;
    alg->B = &B;
    alg->T = NULL;
    alg->maxIter = maxIter;
    alg->tol = tol;

    /* B-positive initialization: deterministic random vectors in V+. */
    srand(123);
    for (uint64_t k = 0; k < sizeSub; k++) {
        for (uint64_t j = 0; j < m; j++) {
            f64 re = (f64)rand() / RAND_MAX - 0.5;
            f64 im = (f64)rand() / RAND_MAX - 0.5;
            c64 val = re + im * I;
            alg->S[j + k * n]     = val;
            alg->S[m + j + k * n] = val;
        }
    }

    z_ilobpcg(alg);

    printf("conv=%lu iter=%lu ", (unsigned long)alg->converged,
           (unsigned long)alg->iter);
    ASSERT(alg->converged == nev);

    /* Check eigenvalues are positive and match (k*pi)^2 */
    for (uint64_t k = 1; k <= nev; k++) {
        f64 exact = (k * PI) * (k * PI);
        ASSERT(alg->eigVals[k - 1] > 0);
        f64 rel_err = fabs(alg->eigVals[k - 1] - exact) / exact;
        ASSERT(rel_err < 0.01);
    }

    z_lobpcg_free(&alg);
    safe_free((void **)&ctx_a);
    safe_free((void **)&ctx_b);
}

/* ================================================================ */
int main(void) {
    printf("Indefinite LOBPCG tests:\n");
    RUN(d_ilobpcg_block_laplacian);
    RUN(z_ilobpcg_block_laplacian);

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}
