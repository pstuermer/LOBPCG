/**
 * @file test_residual.c
 * @brief Test residual computation functions
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

#define TEST_TOLERANCE 1e-12

/* Simple diagonal matrix operator */
typedef struct {
    uint64_t n;
    f64 *diag;
} diag_ctx_t;

void diag_matvec_d(const LinearOperator_d_t *op, const f64 *x, f64 *y) {
    diag_ctx_t *ctx = (diag_ctx_t*)op->ctx;
    for (uint64_t i = 0; i < ctx->n; i++) {
        y[i] = ctx->diag[i] * x[i];
    }
}

void diag_cleanup(linop_ctx_t *ctx) {
    diag_ctx_t *dctx = (diag_ctx_t*)ctx;
    safe_free((void**)&dctx->diag);
    safe_free((void**)&dctx);
}

/* Test get_residual with exact eigenvectors */
int test_get_residual(void) {
    const uint64_t n = 10;
    const uint64_t nev = 3;

    /* Create diagonal matrix with eigenvalues 1, 2, 3, ..., n */
    diag_ctx_t *ctx = xcalloc(1, sizeof(diag_ctx_t));
    ctx->n = n;
    ctx->diag = xcalloc(n, sizeof(f64));
    for (uint64_t i = 0; i < n; i++) {
        ctx->diag[i] = (f64)(i + 1);
    }

    LinearOperator_d_t A;
    A.rows = n;
    A.cols = n;
    A.matvec = diag_matvec_d;
    A.cleanup = diag_cleanup;
    A.ctx = ctx;

    /* Exact eigenvectors (standard basis) and eigenvalues */
    f64 *X = xcalloc(n * nev, sizeof(f64));
    f64 *eigVal = xcalloc(nev, sizeof(f64));
    for (uint64_t i = 0; i < nev; i++) {
        X[i + i*n] = 1.0;  /* e_i */
        eigVal[i] = (f64)(i + 1);
    }

    /* Compute residual */
    f64 *R = xcalloc(n * nev, sizeof(f64));
    f64 *wrk = xcalloc(n * nev, sizeof(f64));

    d_get_residual(n, nev, X, NULL, R, eigVal, wrk, &A, NULL);

    /* Check ||R||_F (should be ~ 0 for exact eigenvectors) */
    f64 R_norm = d_nrm2(n * nev, R);
    printf("  ||R||_F = %.3e (should be < %.3e)\n", R_norm, TEST_TOLERANCE);

    int pass = (R_norm < TEST_TOLERANCE);

    safe_free((void**)&X); safe_free((void**)&eigVal); safe_free((void**)&R); safe_free((void**)&wrk);
    diag_cleanup(ctx);

    return pass;
}

/* Test get_residual_norm */
int test_get_residual_norm(void) {
    const uint64_t n = 10;
    const uint64_t nev = 3;

    /* Create small random residuals */
    f64 *W = xcalloc(n * nev, sizeof(f64));
    for (uint64_t i = 0; i < n * nev; i++) {
        W[i] = 1e-8 * ((f64)rand() / RAND_MAX);
    }

    /* Eigenvalues */
    f64 *eigVals = xcalloc(nev, sizeof(f64));
    for (uint64_t i = 0; i < nev; i++) {
        eigVals[i] = (f64)(i + 1);
    }

    /* Compute residual norms */
    f64 *resNorm = xcalloc(nev, sizeof(f64));
    f64 *wrk1 = xcalloc(n * nev, sizeof(f64));
    f64 *wrk2 = xcalloc(n * nev, sizeof(f64));
    f64 *wrk3 = xcalloc(nev * nev, sizeof(f64));

    f64 ANorm = 10.0;
    f64 BNorm = 1.0;

    d_get_residual_norm(n, nev, nev, W, eigVals, resNorm,
                        wrk1, wrk2, wrk3, ANorm, BNorm, NULL);

    printf("  Residual norms:\n");
    int pass = 1;
    for (uint64_t i = 0; i < nev; i++) {
        printf("    resNorm[%lu] = %.3e\n", (unsigned long)i, resNorm[i]);
        /* Should be very small since W is small */
        if (resNorm[i] > 1e-7) pass = 0;
    }

    safe_free((void**)&W); safe_free((void**)&eigVals); safe_free((void**)&resNorm); safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3);

    return pass;
}

int main(void) {
    printf("Testing residual functions...\n");

    printf("\nTest 1: get_residual with exact eigenvectors\n");
    int test1 = test_get_residual();
    printf("  Result: %s\n", test1 ? "PASS" : "FAIL");

    printf("\nTest 2: get_residual_norm\n");
    int test2 = test_get_residual_norm();
    printf("  Result: %s\n", test2 ? "PASS" : "FAIL");

    if (test1 && test2) {
        printf("\nAll tests PASSED\n");
        return 0;
    } else {
        printf("\nSome tests FAILED\n");
        return 1;
    }
}
