/**
 * @file test_rayleigh_ritz.c
 * @brief Test Rayleigh-Ritz projection
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "types.h"
#include "lobpcg.h"
#include "linop.h"
#include "lobpcg/blas_wrapper.h"

#define TEST_TOLERANCE 1e-10

/* Simple diagonal matrix operator for testing */
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
    free(dctx->diag);
    free(dctx);
}

/* Test rayleigh_ritz on diagonal matrix */
int test_rayleigh_ritz_diagonal(void) {
    const uint64_t n = 20;
    const uint64_t nev = 5;

    /* Create diagonal matrix A with eigenvalues 1, 2, 3, ..., n */
    diag_ctx_t *ctx = malloc(sizeof(diag_ctx_t));
    ctx->n = n;
    ctx->diag = malloc(n * sizeof(f64));
    for (uint64_t i = 0; i < n; i++) {
        ctx->diag[i] = (f64)(i + 1);
    }

    LinearOperator_d_t A;
    A.rows = n;
    A.cols = n;
    A.matvec = diag_matvec_d;
    A.cleanup = diag_cleanup;
    A.ctx = ctx;

    /* Create initial subspace S (n x nev) - should capture first nev eigenvectors */
    f64 *S = calloc(n * nev, sizeof(f64));
    for (uint64_t i = 0; i < nev; i++) {
        S[i + i*n] = 1.0;  /* Standard basis vectors */
    }

    /* Allocate outputs and workspaces */
    f64 *Cx = calloc(nev * nev, sizeof(f64));
    f64 *eigVal = calloc(nev, sizeof(f64));
    f64 *wrk1 = calloc(nev * nev, sizeof(f64));
    f64 *wrk2 = calloc(n * nev, sizeof(f64));
    f64 *wrk3 = calloc(nev * nev, sizeof(f64));

    /* Run Rayleigh-Ritz */
    d_rayleigh_ritz(n, nev, S, Cx, eigVal, wrk1, wrk2, wrk3, &A, NULL);

    /* Check eigenvalues (should be 1, 2, 3, 4, 5) */
    printf("  Computed eigenvalues:\n");
    f64 max_error = 0;
    for (uint64_t i = 0; i < nev; i++) {
        f64 expected = (f64)(i + 1);
        f64 error = fabs(eigVal[i] - expected);
        max_error = (error > max_error) ? error : max_error;
        printf("    λ[%lu] = %.6f (expected %.6f, error = %.3e)\n",
               (unsigned long)i, eigVal[i], expected, error);
    }

    int pass = (max_error < TEST_TOLERANCE);

    free(S); free(Cx); free(eigVal); free(wrk1); free(wrk2); free(wrk3);
    diag_cleanup(ctx);

    return pass;
}

int main(void) {
    printf("Testing rayleigh_ritz...\n");

    printf("\nTest 1: Diagonal matrix eigenvalues\n");
    int test1 = test_rayleigh_ritz_diagonal();
    printf("  Result: %s\n", test1 ? "PASS" : "FAIL");

    if (test1) {
        printf("\nAll tests PASSED\n");
        return 0;
    } else {
        printf("\nSome tests FAILED\n");
        return 1;
    }
}
