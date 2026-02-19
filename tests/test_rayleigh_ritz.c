/**
 * @file test_rayleigh_ritz.c
 * @brief Tests for standard and modified Rayleigh-Ritz procedures (all 4 types)
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

#define TEST_TOL_D 1e-10
#define TEST_TOL_S 1e-4

/* ================================================================
 * Diagonal matvec helpers for each type
 * ================================================================ */

/* --- float --- */
typedef struct { uint64_t n; f32 *diag; } diag_ctx_s_t;

void diag_matvec_s(const LinearOperator_s_t *op, f32 *restrict x, f32 *restrict y) {
    diag_ctx_s_t *ctx = (diag_ctx_s_t *)op->ctx;
    for (uint64_t i = 0; i < ctx->n; i++) y[i] = ctx->diag[i] * x[i];
}

/* --- double --- */
typedef struct { uint64_t n; f64 *diag; } diag_ctx_d_t;

void diag_matvec_d(const LinearOperator_d_t *op, f64 *restrict x, f64 *restrict y) {
    diag_ctx_d_t *ctx = (diag_ctx_d_t *)op->ctx;
    for (uint64_t i = 0; i < ctx->n; i++) y[i] = ctx->diag[i] * x[i];
}

/* --- complex float --- */
typedef struct { uint64_t n; f32 *diag; } diag_ctx_c_t;

void diag_matvec_c(const LinearOperator_c_t *op, c32 *restrict x, c32 *restrict y) {
    diag_ctx_c_t *ctx = (diag_ctx_c_t *)op->ctx;
    for (uint64_t i = 0; i < ctx->n; i++) y[i] = ctx->diag[i] * x[i];
}

/* --- complex double --- */
typedef struct { uint64_t n; f64 *diag; } diag_ctx_z_t;

void diag_matvec_z(const LinearOperator_z_t *op, c64 *restrict x, c64 *restrict y) {
    diag_ctx_z_t *ctx = (diag_ctx_z_t *)op->ctx;
    for (uint64_t i = 0; i < ctx->n; i++) y[i] = ctx->diag[i] * x[i];
}

/* ================================================================
 * Test: standard rayleigh_ritz — double
 * ================================================================ */
int test_rr_d(void) {
    const uint64_t n = 20, nev = 5;

    diag_ctx_d_t *ctx = xcalloc(1, sizeof(diag_ctx_d_t));
    ctx->n = n;
    ctx->diag = xcalloc(n, sizeof(f64));
    for (uint64_t i = 0; i < n; i++) ctx->diag[i] = (f64)(i + 1);

    LinearOperator_d_t A = { .rows = n, .cols = n, .matvec = diag_matvec_d, .ctx = (linop_ctx_t *)ctx };

    f64 *S    = xcalloc(n * nev, sizeof(f64));
    f64 *Cx   = xcalloc(nev * nev, sizeof(f64));
    f64 *eigVal = xcalloc(nev, sizeof(f64));
    f64 *wrk1 = xcalloc(nev * nev, sizeof(f64));
    f64 *wrk2 = xcalloc(n * nev, sizeof(f64));
    f64 *wrk3 = xcalloc(nev * nev, sizeof(f64));

    for (uint64_t i = 0; i < nev; i++) S[i + i*n] = 1.0;

    d_rayleigh_ritz(n, nev, S, Cx, eigVal, wrk1, wrk2, wrk3, &A, NULL);

    f64 max_err = 0;
    for (uint64_t i = 0; i < nev; i++) {
        f64 err = fabs(eigVal[i] - (f64)(i + 1));
        if (err > max_err) max_err = err;
    }

    int pass = (max_err < TEST_TOL_D);
    printf("  d_rayleigh_ritz: max_err = %.3e  %s\n", max_err, pass ? "PASS" : "FAIL");

    safe_free((void**)&S);    safe_free((void**)&Cx);   safe_free((void**)&eigVal);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3);
    safe_free((void**)&ctx->diag); safe_free((void**)&ctx);
    return pass;
}

/* ================================================================
 * Test: standard rayleigh_ritz — float
 * ================================================================ */
int test_rr_s(void) {
    const uint64_t n = 20, nev = 5;

    diag_ctx_s_t *ctx = xcalloc(1, sizeof(diag_ctx_s_t));
    ctx->n = n;
    ctx->diag = xcalloc(n, sizeof(f32));
    for (uint64_t i = 0; i < n; i++) ctx->diag[i] = (f32)(i + 1);

    LinearOperator_s_t A = { .rows = n, .cols = n, .matvec = diag_matvec_s, .ctx = (linop_ctx_t *)ctx };

    f32 *S    = xcalloc(n * nev, sizeof(f32));
    f32 *Cx   = xcalloc(nev * nev, sizeof(f32));
    f32 *eigVal = xcalloc(nev, sizeof(f32));
    f32 *wrk1 = xcalloc(nev * nev, sizeof(f32));
    f32 *wrk2 = xcalloc(n * nev, sizeof(f32));
    f32 *wrk3 = xcalloc(nev * nev, sizeof(f32));

    for (uint64_t i = 0; i < nev; i++) S[i + i*n] = 1.0f;

    s_rayleigh_ritz(n, nev, S, Cx, eigVal, wrk1, wrk2, wrk3, &A, NULL);

    f32 max_err = 0;
    for (uint64_t i = 0; i < nev; i++) {
        f32 err = fabsf(eigVal[i] - (f32)(i + 1));
        if (err > max_err) max_err = err;
    }

    int pass = (max_err < TEST_TOL_S);
    printf("  s_rayleigh_ritz: max_err = %.3e  %s\n", max_err, pass ? "PASS" : "FAIL");

    safe_free((void**)&S);    safe_free((void**)&Cx);   safe_free((void**)&eigVal);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3);
    safe_free((void**)&ctx->diag); safe_free((void**)&ctx);
    return pass;
}

/* ================================================================
 * Test: standard rayleigh_ritz — complex double
 * ================================================================ */
int test_rr_z(void) {
    const uint64_t n = 20, nev = 5;

    diag_ctx_z_t *ctx = xcalloc(1, sizeof(diag_ctx_z_t));
    ctx->n = n;
    ctx->diag = xcalloc(n, sizeof(f64));
    for (uint64_t i = 0; i < n; i++) ctx->diag[i] = (f64)(i + 1);

    LinearOperator_z_t A = { .rows = n, .cols = n, .matvec = diag_matvec_z, .ctx = (linop_ctx_t *)ctx };

    c64 *S    = xcalloc(n * nev, sizeof(c64));
    c64 *Cx   = xcalloc(nev * nev, sizeof(c64));
    f64 *eigVal = xcalloc(nev, sizeof(f64));
    c64 *wrk1 = xcalloc(nev * nev, sizeof(c64));
    c64 *wrk2 = xcalloc(n * nev, sizeof(c64));
    c64 *wrk3 = xcalloc(nev * nev, sizeof(c64));

    for (uint64_t i = 0; i < nev; i++) S[i + i*n] = 1.0 + 0*I;

    z_rayleigh_ritz(n, nev, S, Cx, eigVal, wrk1, wrk2, wrk3, &A, NULL);

    f64 max_err = 0;
    for (uint64_t i = 0; i < nev; i++) {
        f64 err = fabs(eigVal[i] - (f64)(i + 1));
        if (err > max_err) max_err = err;
    }

    int pass = (max_err < TEST_TOL_D);
    printf("  z_rayleigh_ritz: max_err = %.3e  %s\n", max_err, pass ? "PASS" : "FAIL");

    safe_free((void**)&S);    safe_free((void**)&Cx);   safe_free((void**)&eigVal);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3);
    safe_free((void**)&ctx->diag); safe_free((void**)&ctx);
    return pass;
}

/* ================================================================
 * Test: standard rayleigh_ritz — complex float
 * ================================================================ */
int test_rr_c(void) {
    const uint64_t n = 20, nev = 5;

    diag_ctx_c_t *ctx = xcalloc(1, sizeof(diag_ctx_c_t));
    ctx->n = n;
    ctx->diag = xcalloc(n, sizeof(f32));
    for (uint64_t i = 0; i < n; i++) ctx->diag[i] = (f32)(i + 1);

    LinearOperator_c_t A = { .rows = n, .cols = n, .matvec = diag_matvec_c, .ctx = (linop_ctx_t *)ctx };

    c32 *S    = xcalloc(n * nev, sizeof(c32));
    c32 *Cx   = xcalloc(nev * nev, sizeof(c32));
    f32 *eigVal = xcalloc(nev, sizeof(f32));
    c32 *wrk1 = xcalloc(nev * nev, sizeof(c32));
    c32 *wrk2 = xcalloc(n * nev, sizeof(c32));
    c32 *wrk3 = xcalloc(nev * nev, sizeof(c32));

    for (uint64_t i = 0; i < nev; i++) S[i + i*n] = 1.0f + 0*I;

    c_rayleigh_ritz(n, nev, S, Cx, eigVal, wrk1, wrk2, wrk3, &A, NULL);

    f32 max_err = 0;
    for (uint64_t i = 0; i < nev; i++) {
        f32 err = fabsf(eigVal[i] - (f32)(i + 1));
        if (err > max_err) max_err = err;
    }

    int pass = (max_err < TEST_TOL_S);
    printf("  c_rayleigh_ritz: max_err = %.3e  %s\n", max_err, pass ? "PASS" : "FAIL");

    safe_free((void**)&S);    safe_free((void**)&Cx);   safe_free((void**)&eigVal);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3);
    safe_free((void**)&ctx->diag); safe_free((void**)&ctx);
    return pass;
}

/* ================================================================
 * Test: rayleigh_ritz_modified — double, mult=2
 *
 * S = [e1..e_nev | random_cols] (n x 2*nev), mult=2
 * A = diag(1..n)
 * Verify eigenvalues match expected lowest nev eigenvalues
 * Verify Cx reconstruction: X_new = S * Cx gives correct eigenvectors
 * ================================================================ */
int test_rr_modified_d(void) {
    const uint64_t n = 20, nev = 3;
    const uint64_t mult = 2;
    const uint64_t sizeSub = mult * nev;

    diag_ctx_d_t *ctx = xcalloc(1, sizeof(diag_ctx_d_t));
    ctx->n = n;
    ctx->diag = xcalloc(n, sizeof(f64));
    for (uint64_t i = 0; i < n; i++) ctx->diag[i] = (f64)(i + 1);

    LinearOperator_d_t A = { .rows = n, .cols = n, .matvec = diag_matvec_d, .ctx = (linop_ctx_t *)ctx };

    /* S = [e1..e_nev | e_{nev+1}..e_{2*nev}] — standard basis */
    f64 *S    = xcalloc(n * sizeSub, sizeof(f64));
    for (uint64_t j = 0; j < sizeSub; j++) S[j + j*n] = 1.0;

    const uint64_t n_rem = (mult - 1) * nev;
    f64 *Cx   = xcalloc(sizeSub * sizeSub, sizeof(f64));  /* used as temp in step 9 */
    f64 *Cp   = xcalloc(sizeSub * n_rem, sizeof(f64));
    f64 *eigVal = xcalloc(nev, sizeof(f64));
    f64 *wrk1 = xcalloc(sizeSub * sizeSub, sizeof(f64));
    f64 *wrk2 = xcalloc(n * sizeSub, sizeof(f64));
    f64 *wrk3 = xcalloc(sizeSub * sizeSub, sizeof(f64));

    uint8_t useOrtho = 0;

    d_rayleigh_ritz_modified(n, nev, mult, 0, 0, &useOrtho,
                             S, wrk1, wrk2, wrk3, Cx, Cp, eigVal, &A, NULL);

    /* Eigenvalues should be 1, 2, 3 (lowest of diag 1..6) */
    f64 max_err = 0;
    printf("  Modified RR eigenvalues:\n");
    for (uint64_t i = 0; i < nev; i++) {
        f64 expected = (f64)(i + 1);
        f64 err = fabs(eigVal[i] - expected);
        if (err > max_err) max_err = err;
        printf("    λ[%lu] = %.6f (expected %.1f, error = %.3e)\n",
               (unsigned long)i, eigVal[i], expected, err);
    }

    /* Verify X_new = S * Cx produces correct eigenvectors
     * Cx is sizeSub x nev (first nev columns of back-transformed eigenvectors) */
    f64 *X_new = xcalloc(n * nev, sizeof(f64));
    d_gemm_nn(n, nev, sizeSub, 1.0, S, Cx, 0.0, X_new);

    /* Each column of X_new should be a standard basis vector (up to sign) */
    f64 recon_err = 0;
    for (uint64_t j = 0; j < nev; j++) {
        /* Column j should point in direction e_{j+1} (eigenvalue j+1) */
        /* Check that |X_new[j, j]| ≈ 1 and others ≈ 0 */
        f64 col_norm = 0;
        for (uint64_t i = 0; i < n; i++) col_norm += X_new[i + j*n] * X_new[i + j*n];
        col_norm = sqrt(col_norm);
        f64 dominant = fabs(X_new[j + j*n]) / col_norm;
        f64 err = fabs(dominant - 1.0);
        if (err > recon_err) recon_err = err;
    }
    printf("  Cx reconstruction error: %.3e\n", recon_err);

    int pass = (max_err < TEST_TOL_D) && (recon_err < TEST_TOL_D) && (0 != useOrtho || 1);
    printf("  useOrtho = %d (should be 0 for well-conditioned input)\n", useOrtho);
    /* useOrtho should remain 0 if condition is fine */
    pass = pass && (0 == useOrtho);
    printf("  d_rayleigh_ritz_modified (mult=2): %s\n", pass ? "PASS" : "FAIL");

    safe_free((void**)&S);    safe_free((void**)&Cx);   safe_free((void**)&Cp);
    safe_free((void**)&eigVal);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3);
    safe_free((void**)&X_new);
    safe_free((void**)&ctx->diag); safe_free((void**)&ctx);
    return pass;
}

/* ================================================================
 * Test: rayleigh_ritz_modified — double, mult=3
 * ================================================================ */
int test_rr_modified_d_mult3(void) {
    const uint64_t n = 20, nev = 3;
    const uint64_t mult = 3;
    const uint64_t sizeSub = mult * nev;

    diag_ctx_d_t *ctx = xcalloc(1, sizeof(diag_ctx_d_t));
    ctx->n = n;
    ctx->diag = xcalloc(n, sizeof(f64));
    for (uint64_t i = 0; i < n; i++) ctx->diag[i] = (f64)(i + 1);

    LinearOperator_d_t A = { .rows = n, .cols = n, .matvec = diag_matvec_d, .ctx = (linop_ctx_t *)ctx };

    /* S = [e1..e_nev | e_{nev+1}..e_{2*nev} | e_{2*nev+1}..e_{3*nev}] */
    f64 *S    = xcalloc(n * sizeSub, sizeof(f64));
    for (uint64_t j = 0; j < sizeSub; j++) S[j + j*n] = 1.0;

    const uint64_t n_rem = (mult - 1) * nev;
    f64 *Cx   = xcalloc(sizeSub * sizeSub, sizeof(f64));
    f64 *Cp   = xcalloc(sizeSub * n_rem, sizeof(f64));
    f64 *eigVal = xcalloc(nev, sizeof(f64));
    f64 *wrk1 = xcalloc(sizeSub * sizeSub, sizeof(f64));
    f64 *wrk2 = xcalloc(n * sizeSub, sizeof(f64));
    f64 *wrk3 = xcalloc(sizeSub * sizeSub, sizeof(f64));

    uint8_t useOrtho = 0;

    d_rayleigh_ritz_modified(n, nev, mult, 0, 0, &useOrtho,
                             S, wrk1, wrk2, wrk3, Cx, Cp, eigVal, &A, NULL);

    /* Eigenvalues should be 1, 2, 3 (lowest of diag 1..9) */
    f64 max_err = 0;
    for (uint64_t i = 0; i < nev; i++) {
        f64 err = fabs(eigVal[i] - (f64)(i + 1));
        if (err > max_err) max_err = err;
    }

    int pass = (max_err < TEST_TOL_D) && (0 == useOrtho);
    printf("  d_rayleigh_ritz_modified (mult=3): max_err = %.3e  %s\n",
           max_err, pass ? "PASS" : "FAIL");

    safe_free((void**)&S);    safe_free((void**)&Cx);   safe_free((void**)&Cp);
    safe_free((void**)&eigVal);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3);
    safe_free((void**)&ctx->diag); safe_free((void**)&ctx);
    return pass;
}

/* ================================================================
 * Test: rayleigh_ritz_modified — complex double, mult=2
 * ================================================================ */
int test_rr_modified_z(void) {
    const uint64_t n = 20, nev = 3;
    const uint64_t mult = 2;
    const uint64_t sizeSub = mult * nev;

    diag_ctx_z_t *ctx = xcalloc(1, sizeof(diag_ctx_z_t));
    ctx->n = n;
    ctx->diag = xcalloc(n, sizeof(f64));
    for (uint64_t i = 0; i < n; i++) ctx->diag[i] = (f64)(i + 1);

    LinearOperator_z_t A = { .rows = n, .cols = n, .matvec = diag_matvec_z, .ctx = (linop_ctx_t *)ctx };

    c64 *S    = xcalloc(n * sizeSub, sizeof(c64));
    for (uint64_t j = 0; j < sizeSub; j++) S[j + j*n] = 1.0 + 0*I;

    const uint64_t n_rem = (mult - 1) * nev;
    c64 *Cx   = xcalloc(sizeSub * sizeSub, sizeof(c64));
    c64 *Cp   = xcalloc(sizeSub * n_rem, sizeof(c64));
    f64 *eigVal = xcalloc(nev, sizeof(f64));
    c64 *wrk1 = xcalloc(sizeSub * sizeSub, sizeof(c64));
    c64 *wrk2 = xcalloc(n * sizeSub, sizeof(c64));
    c64 *wrk3 = xcalloc(sizeSub * sizeSub, sizeof(c64));

    uint8_t useOrtho = 0;

    z_rayleigh_ritz_modified(n, nev, mult, 0, 0, &useOrtho,
                             S, wrk1, wrk2, wrk3, Cx, Cp, eigVal, &A, NULL);

    f64 max_err = 0;
    for (uint64_t i = 0; i < nev; i++) {
        f64 err = fabs(eigVal[i] - (f64)(i + 1));
        if (err > max_err) max_err = err;
    }

    int pass = (max_err < TEST_TOL_D) && (0 == useOrtho);
    printf("  z_rayleigh_ritz_modified (mult=2): max_err = %.3e  %s\n",
           max_err, pass ? "PASS" : "FAIL");

    safe_free((void**)&S);    safe_free((void**)&Cx);   safe_free((void**)&Cp);
    safe_free((void**)&eigVal);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3);
    safe_free((void**)&ctx->diag); safe_free((void**)&ctx);
    return pass;
}

/* ================================================================
 * Test: rayleigh_ritz with B operator (double)
 * A = diag(1..n), B = 2*I  =>  generalized eigvals = A/B = 0.5, 1, 1.5, ...
 * ================================================================ */

typedef struct { uint64_t n; f64 scale; } scale_ctx_d_t;

void scale_matvec_d(const LinearOperator_d_t *op, f64 *restrict x, f64 *restrict y) {
    scale_ctx_d_t *ctx = (scale_ctx_d_t *)op->ctx;
    for (uint64_t i = 0; i < ctx->n; i++) y[i] = ctx->scale * x[i];
}

int test_rr_d_with_B(void) {
    const uint64_t n = 20, nev = 5;

    diag_ctx_d_t *ctxA = xcalloc(1, sizeof(diag_ctx_d_t));
    ctxA->n = n;
    ctxA->diag = xcalloc(n, sizeof(f64));
    for (uint64_t i = 0; i < n; i++) ctxA->diag[i] = (f64)(i + 1);

    scale_ctx_d_t *ctxB = xcalloc(1, sizeof(scale_ctx_d_t));
    ctxB->n = n;
    ctxB->scale = 2.0;

    LinearOperator_d_t A = { .rows = n, .cols = n, .matvec = diag_matvec_d, .ctx = (linop_ctx_t *)ctxA };
    LinearOperator_d_t B = { .rows = n, .cols = n, .matvec = scale_matvec_d, .ctx = (linop_ctx_t *)ctxB };

    f64 *S    = xcalloc(n * nev, sizeof(f64));
    f64 *Cx   = xcalloc(nev * nev, sizeof(f64));
    f64 *eigVal = xcalloc(nev, sizeof(f64));
    f64 *wrk1 = xcalloc(nev * nev, sizeof(f64));
    f64 *wrk2 = xcalloc(n * nev, sizeof(f64));
    f64 *wrk3 = xcalloc(nev * nev, sizeof(f64));

    for (uint64_t i = 0; i < nev; i++) S[i + i*n] = 1.0;

    d_rayleigh_ritz(n, nev, S, Cx, eigVal, wrk1, wrk2, wrk3, &A, &B);

    /* Generalized eigenvalues: λ_i = (i+1)/2 */
    f64 max_err = 0;
    for (uint64_t i = 0; i < nev; i++) {
        f64 expected = (f64)(i + 1) / 2.0;
        f64 err = fabs(eigVal[i] - expected);
        if (err > max_err) max_err = err;
    }

    int pass = (max_err < TEST_TOL_D);
    printf("  d_rayleigh_ritz (B=2I): max_err = %.3e  %s\n", max_err, pass ? "PASS" : "FAIL");

    safe_free((void**)&S);    safe_free((void**)&Cx);   safe_free((void**)&eigVal);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3);
    safe_free((void**)&ctxA->diag); safe_free((void**)&ctxA);
    safe_free((void**)&ctxB);
    return pass;
}

/* ================================================================ */
int main(void) {
    int pass = 1;
    int result;

    printf("=== Standard Rayleigh-Ritz ===\n");

    result = test_rr_d();    pass &= result;
    result = test_rr_s();    pass &= result;
    result = test_rr_z();    pass &= result;
    result = test_rr_c();    pass &= result;
    result = test_rr_d_with_B(); pass &= result;

    printf("\n=== Modified Rayleigh-Ritz ===\n");

    result = test_rr_modified_d();       pass &= result;
    result = test_rr_modified_d_mult3(); pass &= result;
    result = test_rr_modified_z();       pass &= result;

    printf("\n%s\n", pass ? "All tests PASSED" : "Some tests FAILED");
    return pass ? 0 : 1;
}
