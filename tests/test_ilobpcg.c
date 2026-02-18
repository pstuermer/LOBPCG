/**
 * @file test_ilobpcg.c
 * @brief Integration test for indefinite LOBPCG solver
 *
 * Test problem: diagonal A and indefinite diagonal B
 *   A = diag(1, 2, 3, ..., n)
 *   B = diag(+1, ..., +1, -1, ..., -1)  (first n/2 positive, rest negative)
 *
 * Generalized eigenvalue problem: A*x = lambda*B*x
 * For diagonal case: lambda_i = A_ii / B_ii
 *   Positive signature (B_ii = +1): lambda = 1, 2, ..., n/2
 *   Negative signature (B_ii = -1): lambda = -(n/2+1), -(n/2+2), ..., -n
 *
 * After sorting (positive ascending, negative descending):
 *   First nev eigenvalues should be {1, 2, 3, ...} with positive signature
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

/* Diagonal A operator: A = diag(1, 2, 3, ..., n) */
typedef struct {
    uint64_t n;
} diag_ctx_t;

void diag_A_matvec_d(const LinearOperator_d_t *op, const f64 *x, f64 *y) {
    diag_ctx_t *ctx = (diag_ctx_t *)op->ctx;
    for (uint64_t i = 0; i < ctx->n; i++) {
        y[i] = (f64)(i + 1) * x[i];
    }
}

/* Indefinite diagonal B operator: B = diag(+1,...,+1,-1,...,-1) */
void diag_B_matvec_d(const LinearOperator_d_t *op, const f64 *x, f64 *y) {
    diag_ctx_t *ctx = (diag_ctx_t *)op->ctx;
    uint64_t half = ctx->n / 2;
    for (uint64_t i = 0; i < ctx->n; i++) {
        y[i] = (i < half) ? x[i] : -x[i];
    }
}

int test_ilobpcg_diagonal(void) {
    const uint64_t n = 100;
    const uint64_t nev = 3;
    const uint64_t maxIter = 200;
    const f64 tol = 1e-6;

    /* Create diagonal A */
    diag_ctx_t *ctx_a = xcalloc(1, sizeof(diag_ctx_t));
    ctx_a->n = n;

    LinearOperator_d_t A;
    A.rows = n;
    A.cols = n;
    A.matvec = diag_A_matvec_d;
    A.cleanup = NULL;
    A.ctx = (linop_ctx_t *)ctx_a;

    /* Create indefinite diagonal B */
    diag_ctx_t *ctx_b = xcalloc(1, sizeof(diag_ctx_t));
    ctx_b->n = n;

    LinearOperator_d_t B;
    B.rows = n;
    B.cols = n;
    B.matvec = diag_B_matvec_d;
    B.cleanup = NULL;
    B.ctx = (linop_ctx_t *)ctx_b;

    /* Allocate LOBPCG state */
    d_lobpcg_t *alg = xcalloc(1, sizeof(d_lobpcg_t));
    alg->size = n;
    alg->nev = nev;
    alg->sizeSub = nev;
    alg->maxIter = maxIter;
    alg->tol = tol;
    alg->A = &A;
    alg->B = &B;
    alg->T = NULL;

    /* Allocate workspaces */
    alg->S = xcalloc(n * 3 * nev, sizeof(f64));
    alg->AS = xcalloc(n * 3 * nev, sizeof(f64));
    alg->Cx = xcalloc(3 * nev * nev, sizeof(f64));
    alg->Cp = xcalloc(3 * nev * 2 * nev, sizeof(f64));
    alg->eigVals = xcalloc(nev, sizeof(f64));
    alg->resNorm = xcalloc(nev, sizeof(f64));
    alg->signature = xcalloc(3 * nev, sizeof(int8_t));
    alg->wrk1 = xcalloc(n * 3 * nev, sizeof(f64));
    alg->wrk2 = xcalloc(n * 3 * nev, sizeof(f64));
    alg->wrk3 = xcalloc(3 * nev * 3 * nev, sizeof(f64));
    alg->wrk4 = xcalloc(n * 3 * nev, sizeof(f64));

    /* Run indefinite LOBPCG */
    printf("  Running iLOBPCG (n=%lu, nev=%lu, maxIter=%lu, tol=%.3e)...\n",
           (unsigned long)n, (unsigned long)nev, (unsigned long)maxIter, tol);
    d_ilobpcg(alg);

    printf("  Converged %lu/%lu eigenpairs in %lu iterations\n",
           (unsigned long)alg->converged, (unsigned long)nev, (unsigned long)alg->iter);

    /* Expected: smallest positive eigenvalues are 1, 2, 3 */
    printf("  Eigenvalues:\n");
    f64 max_error = 0;
    for (uint64_t k = 0; k < nev; k++) {
        f64 lambda_computed = alg->eigVals[k];
        f64 lambda_exact = (f64)(k + 1);
        f64 error = fabs(lambda_computed - lambda_exact) / lambda_exact;
        max_error = (error > max_error) ? error : max_error;
        printf("    lambda[%lu] = %.6f (exact: %.1f, rel error: %.3e, sig: %d)\n",
               (unsigned long)k, lambda_computed, lambda_exact, error,
               (int)alg->signature[k]);
    }

    printf("  Residual norms:\n");
    for (uint64_t i = 0; i < nev; i++) {
        printf("    resNorm[%lu] = %.3e\n", (unsigned long)i, alg->resNorm[i]);
    }

    int pass = (max_error < 0.01) && (alg->converged == nev);

    /* Cleanup */
    safe_free((void **)&alg->S);
    safe_free((void **)&alg->AS);
    safe_free((void **)&alg->Cx);
    safe_free((void **)&alg->Cp);
    safe_free((void **)&alg->eigVals);
    safe_free((void **)&alg->resNorm);
    safe_free((void **)&alg->signature);
    safe_free((void **)&alg->wrk1);
    safe_free((void **)&alg->wrk2);
    safe_free((void **)&alg->wrk3);
    safe_free((void **)&alg->wrk4);
    safe_free((void **)&alg);
    safe_free((void **)&ctx_a);
    safe_free((void **)&ctx_b);

    return pass;
}

int main(void) {
    printf("Testing indefinite LOBPCG solver...\n");

    printf("\nTest 1: Diagonal indefinite eigenvalue problem\n");
    int test1 = test_ilobpcg_diagonal();
    printf("  Result: %s\n", test1 ? "PASS" : "FAIL");

    if (test1) {
        printf("\nAll tests PASSED\n");
        return 0;
    } else {
        printf("\nSome tests FAILED\n");
        return 1;
    }
}
