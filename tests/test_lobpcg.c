/**
 * @file test_lobpcg.c
 * @brief Integration test for main LOBPCG solver
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "types.h"
#include "lobpcg.h"
#include "linop.h"
#include "lobpcg/blas_wrapper.h"
#include "lobpcg/memory.h"

#define PI 3.14159265358979323846

/* 1D Laplacian operator: -d^2/dx^2 with Dirichlet BC
 * Discretized as tridiagonal: [-1, 2, -1] / h^2
 * Eigenvalues: λ_k = 4/h^2 * sin^2(k*pi*h/2) ≈ (k*pi)^2 for k=1,2,...
 */
typedef struct {
    uint64_t n;
    f64 h;
    f64 h2_inv;
} laplacian_ctx_t;

void laplacian_matvec_d(const LinearOperator_d_t *op, const f64 *x, f64 *y) {
    laplacian_ctx_t *ctx = (laplacian_ctx_t*)op->ctx;
    uint64_t n = ctx->n;
    f64 c = ctx->h2_inv;

    y[0] = c * (2.0 * x[0] - x[1]);
    for (uint64_t i = 1; i < n - 1; i++) {
        y[i] = c * (-x[i-1] + 2.0 * x[i] - x[i+1]);
    }
    y[n-1] = c * (-x[n-2] + 2.0 * x[n-1]);
}

void laplacian_cleanup(linop_ctx_t *ctx) {
    free(ctx);
}

/* Test LOBPCG on 1D Laplacian */
int test_lobpcg_laplacian(void) {
    const uint64_t n = 100;
    const uint64_t nev = 5;
    const uint64_t maxIter = 100;
    const f64 tol = 1e-6;

    /* Create 1D Laplacian */
    f64 L = 1.0;
    f64 h = L / (n + 1);
    laplacian_ctx_t *ctx = malloc(sizeof(laplacian_ctx_t));
    ctx->n = n;
    ctx->h = h;
    ctx->h2_inv = 1.0 / (h * h);

    LinearOperator_d_t A;
    A.rows = n;
    A.cols = n;
    A.matvec = laplacian_matvec_d;
    A.cleanup = laplacian_cleanup;
    A.ctx = ctx;

    /* Allocate LOBPCG state */
    d_lobpcg_t *alg = xcalloc(1, sizeof(d_lobpcg_t));
    alg->size = n;
    alg->nev = nev;
    alg->sizeSub = nev;
    alg->maxIter = maxIter;
    alg->tol = tol;
    alg->A = &A;
    alg->B = NULL;
    alg->T = NULL;

    /* Allocate workspaces */
    alg->S = xcalloc(n * 3 * nev, sizeof(f64));  /* [X, P, W] */
    alg->Cx = xcalloc(3 * nev * nev, sizeof(f64));
    alg->Cp = xcalloc(3 * nev * nev, sizeof(f64));
    alg->eigVals = xcalloc(nev, sizeof(f64));
    alg->resNorm = xcalloc(nev, sizeof(f64));
    alg->wrk1 = xcalloc(n * 3 * nev, sizeof(f64));
    alg->wrk2 = xcalloc(n * 3 * nev, sizeof(f64));
    alg->wrk3 = xcalloc(3 * nev * 3 * nev, sizeof(f64));
    alg->wrk4 = xcalloc(n * 3 * nev, sizeof(f64));

    /* Initialize with random guess (S is zero, will be filled in lobpcg) */

    /* Run LOBPCG */
    printf("  Running LOBPCG (n=%lu, nev=%lu, maxIter=%lu, tol=%.3e)...\n",
           (unsigned long)n, (unsigned long)nev, (unsigned long)maxIter, tol);
    d_lobpcg(alg);

    printf("  Converged %lu/%lu eigenpairs in %lu iterations\n",
           (unsigned long)alg->converged, (unsigned long)nev, (unsigned long)alg->iter);

    /* Check eigenvalues against analytical formula: λ_k ≈ (k*π)^2 */
    printf("  Eigenvalues:\n");
    f64 max_error = 0;
    for (uint64_t k = 1; k <= nev; k++) {
        f64 lambda_computed = alg->eigVals[k-1];
        f64 lambda_exact = (k * PI) * (k * PI);
        f64 error = fabs(lambda_computed - lambda_exact) / lambda_exact;
        max_error = (error > max_error) ? error : max_error;
        printf("    λ[%lu] = %.6f (exact: %.6f, rel error: %.3e)\n",
               (unsigned long)k, lambda_computed, lambda_exact, error);
    }

    printf("  Residual norms:\n");
    for (uint64_t i = 0; i < nev; i++) {
        printf("    resNorm[%lu] = %.3e\n", (unsigned long)i, alg->resNorm[i]);
    }

    int pass = (max_error < 0.01) && (alg->converged == nev);  /* 1% relative error */

    /* Cleanup */
    safe_free((void**)&alg->S);
    safe_free((void**)&alg->Cx);
    safe_free((void**)&alg->Cp);
    safe_free((void**)&alg->eigVals);
    safe_free((void**)&alg->resNorm);
    safe_free((void**)&alg->wrk1);
    safe_free((void**)&alg->wrk2);
    safe_free((void**)&alg->wrk3);
    safe_free((void**)&alg->wrk4);
    safe_free((void**)&alg);
    laplacian_cleanup(ctx);

    return pass;
}

int main(void) {
    printf("Testing LOBPCG solver...\n");

    printf("\nTest 1: 1D Laplacian eigenvalue problem\n");
    int test1 = test_lobpcg_laplacian();
    printf("  Result: %s\n", test1 ? "PASS" : "FAIL");

    if (test1) {
        printf("\nAll tests PASSED\n");
        return 0;
    } else {
        printf("\nSome tests FAILED\n");
        return 1;
    }
}
