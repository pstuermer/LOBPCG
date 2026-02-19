/**
 * @file test_indefinite_rr.c
 * @brief Tests for indefinite Rayleigh-Ritz procedures (all 4 types)
 *
 * Test problem: diagonal A and diagonal indefinite B
 *   A = diag(1, 2, 3, 4, 5, 6)
 *   B = diag(+1, +1, +1, -1, -1, -1)
 *
 * Generalized eigenvalues: lambda_i = A_ii / B_ii
 *   = {1, 2, 3, -4, -5, -6}
 *
 * After sorting (positive ascending, negative descending):
 *   = {1, 2, 3, -4, -5, -6}
 * Signature: {+1, +1, +1, -1, -1, -1}
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "types.h"
#include "lobpcg.h"
#include "lobpcg/linop.h"
#include "lobpcg/blas_wrapper.h"
#include "lobpcg/memory.h"

#define TEST_TOL_D 1e-10
#define TEST_TOL_S 1e-4

/* ================================================================
 * Diagonal matvec helpers
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
 * Test: indefinite_rayleigh_ritz — double
 * ================================================================ */
int test_indef_rr_d(void) {
    const uint64_t n = 6;

    /* A = diag(1,2,3,4,5,6) */
    diag_ctx_d_t *ctxA = xcalloc(1, sizeof(diag_ctx_d_t));
    ctxA->n = n;
    ctxA->diag = xcalloc(n, sizeof(f64));
    for (uint64_t i = 0; i < n; i++) ctxA->diag[i] = (f64)(i + 1);

    /* B = diag(+1,+1,+1,-1,-1,-1) */
    diag_ctx_d_t *ctxB = xcalloc(1, sizeof(diag_ctx_d_t));
    ctxB->n = n;
    ctxB->diag = xcalloc(n, sizeof(f64));
    for (uint64_t i = 0; i < n; i++)
        ctxB->diag[i] = (i < n / 2) ? 1.0 : -1.0;

    LinearOperator_d_t A = { .rows = n, .cols = n, .matvec = diag_matvec_d,
                             .ctx = (linop_ctx_t *)ctxA };
    LinearOperator_d_t B = { .rows = n, .cols = n, .matvec = diag_matvec_d,
                             .ctx = (linop_ctx_t *)ctxB };

    /* S = identity (eigenvectors are standard basis) */
    f64 *S    = xcalloc(n * n, sizeof(f64));
    f64 *Cx   = xcalloc(n * n, sizeof(f64));
    f64 *eigVal = xcalloc(n, sizeof(f64));
    int8_t *sig = xcalloc(n, sizeof(int8_t));
    f64 *wrk1 = xcalloc(n * n, sizeof(f64));
    f64 *wrk2 = xcalloc(n * n, sizeof(f64));
    f64 *wrk3 = xcalloc(n * n, sizeof(f64));
    f64 *wrk4 = xcalloc(n * n, sizeof(f64));

    for (uint64_t i = 0; i < n; i++) S[i + i * n] = 1.0;

    d_indefinite_rayleigh_ritz(n, n, S, Cx, eigVal, sig,
                                wrk1, wrk2, wrk3, wrk4, &A, &B);

    /* Expected sorted: {1, 2, 3, -4, -5, -6} with sig {+1,+1,+1,-1,-1,-1} */
    f64 expected_eig[] = {1.0, 2.0, 3.0, -4.0, -5.0, -6.0};
    int8_t expected_sig[] = {1, 1, 1, -1, -1, -1};

    f64 max_err = 0;
    int sig_ok = 1;
    printf("  d_indefinite_rayleigh_ritz:\n");
    for (uint64_t i = 0; i < n; i++) {
        f64 err = fabs(eigVal[i] - expected_eig[i]);
        if (err > max_err) max_err = err;
        if (sig[i] != expected_sig[i]) sig_ok = 0;
        printf("    λ[%lu] = %8.4f (exp %6.1f) sig=%+d (exp %+d)\n",
               (unsigned long)i, eigVal[i], expected_eig[i], sig[i], expected_sig[i]);
    }

    int pass = (max_err < TEST_TOL_D) && sig_ok;
    printf("    max_err = %.3e  sig_ok = %d  %s\n", max_err, sig_ok,
           pass ? "PASS" : "FAIL");

    safe_free((void**)&S);    safe_free((void**)&Cx);
    safe_free((void**)&eigVal); safe_free((void**)&sig);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2);
    safe_free((void**)&wrk3); safe_free((void**)&wrk4);
    safe_free((void**)&ctxA->diag); safe_free((void**)&ctxA);
    safe_free((void**)&ctxB->diag); safe_free((void**)&ctxB);
    return pass;
}

/* ================================================================
 * Test: indefinite_rayleigh_ritz — float
 * ================================================================ */
int test_indef_rr_s(void) {
    const uint64_t n = 6;

    diag_ctx_s_t *ctxA = xcalloc(1, sizeof(diag_ctx_s_t));
    ctxA->n = n;
    ctxA->diag = xcalloc(n, sizeof(f32));
    for (uint64_t i = 0; i < n; i++) ctxA->diag[i] = (f32)(i + 1);

    diag_ctx_s_t *ctxB = xcalloc(1, sizeof(diag_ctx_s_t));
    ctxB->n = n;
    ctxB->diag = xcalloc(n, sizeof(f32));
    for (uint64_t i = 0; i < n; i++)
        ctxB->diag[i] = (i < n / 2) ? 1.0f : -1.0f;

    LinearOperator_s_t A = { .rows = n, .cols = n, .matvec = diag_matvec_s,
                             .ctx = (linop_ctx_t *)ctxA };
    LinearOperator_s_t B = { .rows = n, .cols = n, .matvec = diag_matvec_s,
                             .ctx = (linop_ctx_t *)ctxB };

    f32 *S    = xcalloc(n * n, sizeof(f32));
    f32 *Cx   = xcalloc(n * n, sizeof(f32));
    f32 *eigVal = xcalloc(n, sizeof(f32));
    int8_t *sig = xcalloc(n, sizeof(int8_t));
    f32 *wrk1 = xcalloc(n * n, sizeof(f32));
    f32 *wrk2 = xcalloc(n * n, sizeof(f32));
    f32 *wrk3 = xcalloc(n * n, sizeof(f32));
    f32 *wrk4 = xcalloc(n * n, sizeof(f32));

    for (uint64_t i = 0; i < n; i++) S[i + i * n] = 1.0f;

    s_indefinite_rayleigh_ritz(n, n, S, Cx, eigVal, sig,
                                wrk1, wrk2, wrk3, wrk4, &A, &B);

    f32 expected_eig[] = {1.0f, 2.0f, 3.0f, -4.0f, -5.0f, -6.0f};
    int8_t expected_sig[] = {1, 1, 1, -1, -1, -1};

    f32 max_err = 0;
    int sig_ok = 1;
    for (uint64_t i = 0; i < n; i++) {
        f32 err = fabsf(eigVal[i] - expected_eig[i]);
        if (err > max_err) max_err = err;
        if (sig[i] != expected_sig[i]) sig_ok = 0;
    }

    int pass = (max_err < TEST_TOL_S) && sig_ok;
    printf("  s_indefinite_rayleigh_ritz: max_err = %.3e  sig_ok = %d  %s\n",
           max_err, sig_ok, pass ? "PASS" : "FAIL");

    safe_free((void**)&S);    safe_free((void**)&Cx);
    safe_free((void**)&eigVal); safe_free((void**)&sig);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2);
    safe_free((void**)&wrk3); safe_free((void**)&wrk4);
    safe_free((void**)&ctxA->diag); safe_free((void**)&ctxA);
    safe_free((void**)&ctxB->diag); safe_free((void**)&ctxB);
    return pass;
}

/* ================================================================
 * Test: indefinite_rayleigh_ritz — complex double
 * ================================================================ */
int test_indef_rr_z(void) {
    const uint64_t n = 6;

    diag_ctx_z_t *ctxA = xcalloc(1, sizeof(diag_ctx_z_t));
    ctxA->n = n;
    ctxA->diag = xcalloc(n, sizeof(f64));
    for (uint64_t i = 0; i < n; i++) ctxA->diag[i] = (f64)(i + 1);

    diag_ctx_z_t *ctxB = xcalloc(1, sizeof(diag_ctx_z_t));
    ctxB->n = n;
    ctxB->diag = xcalloc(n, sizeof(f64));
    for (uint64_t i = 0; i < n; i++)
        ctxB->diag[i] = (i < n / 2) ? 1.0 : -1.0;

    LinearOperator_z_t A = { .rows = n, .cols = n, .matvec = diag_matvec_z,
                             .ctx = (linop_ctx_t *)ctxA };
    LinearOperator_z_t B = { .rows = n, .cols = n, .matvec = diag_matvec_z,
                             .ctx = (linop_ctx_t *)ctxB };

    c64 *S    = xcalloc(n * n, sizeof(c64));
    c64 *Cx   = xcalloc(n * n, sizeof(c64));
    f64 *eigVal = xcalloc(n, sizeof(f64));
    int8_t *sig = xcalloc(n, sizeof(int8_t));
    c64 *wrk1 = xcalloc(n * n, sizeof(c64));
    c64 *wrk2 = xcalloc(n * n, sizeof(c64));
    c64 *wrk3 = xcalloc(n * n, sizeof(c64));
    c64 *wrk4 = xcalloc(n * n, sizeof(c64));

    for (uint64_t i = 0; i < n; i++) S[i + i * n] = 1.0 + 0 * I;

    z_indefinite_rayleigh_ritz(n, n, S, Cx, eigVal, sig,
                                wrk1, wrk2, wrk3, wrk4, &A, &B);

    f64 expected_eig[] = {1.0, 2.0, 3.0, -4.0, -5.0, -6.0};
    int8_t expected_sig[] = {1, 1, 1, -1, -1, -1};

    f64 max_err = 0;
    int sig_ok = 1;
    for (uint64_t i = 0; i < n; i++) {
        f64 err = fabs(eigVal[i] - expected_eig[i]);
        if (err > max_err) max_err = err;
        if (sig[i] != expected_sig[i]) sig_ok = 0;
    }

    int pass = (max_err < TEST_TOL_D) && sig_ok;
    printf("  z_indefinite_rayleigh_ritz: max_err = %.3e  sig_ok = %d  %s\n",
           max_err, sig_ok, pass ? "PASS" : "FAIL");

    safe_free((void**)&S);    safe_free((void**)&Cx);
    safe_free((void**)&eigVal); safe_free((void**)&sig);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2);
    safe_free((void**)&wrk3); safe_free((void**)&wrk4);
    safe_free((void**)&ctxA->diag); safe_free((void**)&ctxA);
    safe_free((void**)&ctxB->diag); safe_free((void**)&ctxB);
    return pass;
}

/* ================================================================
 * Test: indefinite_rayleigh_ritz — complex float
 * ================================================================ */
int test_indef_rr_c(void) {
    const uint64_t n = 6;

    diag_ctx_c_t *ctxA = xcalloc(1, sizeof(diag_ctx_c_t));
    ctxA->n = n;
    ctxA->diag = xcalloc(n, sizeof(f32));
    for (uint64_t i = 0; i < n; i++) ctxA->diag[i] = (f32)(i + 1);

    diag_ctx_c_t *ctxB = xcalloc(1, sizeof(diag_ctx_c_t));
    ctxB->n = n;
    ctxB->diag = xcalloc(n, sizeof(f32));
    for (uint64_t i = 0; i < n; i++)
        ctxB->diag[i] = (i < n / 2) ? 1.0f : -1.0f;

    LinearOperator_c_t A = { .rows = n, .cols = n, .matvec = diag_matvec_c,
                             .ctx = (linop_ctx_t *)ctxA };
    LinearOperator_c_t B = { .rows = n, .cols = n, .matvec = diag_matvec_c,
                             .ctx = (linop_ctx_t *)ctxB };

    c32 *S    = xcalloc(n * n, sizeof(c32));
    c32 *Cx   = xcalloc(n * n, sizeof(c32));
    f32 *eigVal = xcalloc(n, sizeof(f32));
    int8_t *sig = xcalloc(n, sizeof(int8_t));
    c32 *wrk1 = xcalloc(n * n, sizeof(c32));
    c32 *wrk2 = xcalloc(n * n, sizeof(c32));
    c32 *wrk3 = xcalloc(n * n, sizeof(c32));
    c32 *wrk4 = xcalloc(n * n, sizeof(c32));

    for (uint64_t i = 0; i < n; i++) S[i + i * n] = 1.0f + 0 * I;

    c_indefinite_rayleigh_ritz(n, n, S, Cx, eigVal, sig,
                                wrk1, wrk2, wrk3, wrk4, &A, &B);

    f32 expected_eig[] = {1.0f, 2.0f, 3.0f, -4.0f, -5.0f, -6.0f};
    int8_t expected_sig[] = {1, 1, 1, -1, -1, -1};

    f32 max_err = 0;
    int sig_ok = 1;
    for (uint64_t i = 0; i < n; i++) {
        f32 err = fabsf(eigVal[i] - expected_eig[i]);
        if (err > max_err) max_err = err;
        if (sig[i] != expected_sig[i]) sig_ok = 0;
    }

    int pass = (max_err < TEST_TOL_S) && sig_ok;
    printf("  c_indefinite_rayleigh_ritz: max_err = %.3e  sig_ok = %d  %s\n",
           max_err, sig_ok, pass ? "PASS" : "FAIL");

    safe_free((void**)&S);    safe_free((void**)&Cx);
    safe_free((void**)&eigVal); safe_free((void**)&sig);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2);
    safe_free((void**)&wrk3); safe_free((void**)&wrk4);
    safe_free((void**)&ctxA->diag); safe_free((void**)&ctxA);
    safe_free((void**)&ctxB->diag); safe_free((void**)&ctxB);
    return pass;
}

/* ================================================================
 * Test: indefinite_rayleigh_ritz_modified — double, mult=2
 *
 * A = diag(1,2,3,4,5,6), B = diag(+1,+1,+1,-1,-1,-1)
 * S = identity, nev=3, mult=2 => sizeSub=6
 * Expected: first 3 eigenvalues = {1, 2, 3} (positive sector)
 * ================================================================ */
int test_indef_rr_modified_d(void) {
    const uint64_t n = 6, nev = 3;
    const uint64_t mult = 2;
    const uint64_t sizeSub = mult * nev;

    diag_ctx_d_t *ctxA = xcalloc(1, sizeof(diag_ctx_d_t));
    ctxA->n = n;
    ctxA->diag = xcalloc(n, sizeof(f64));
    for (uint64_t i = 0; i < n; i++) ctxA->diag[i] = (f64)(i + 1);

    diag_ctx_d_t *ctxB = xcalloc(1, sizeof(diag_ctx_d_t));
    ctxB->n = n;
    ctxB->diag = xcalloc(n, sizeof(f64));
    for (uint64_t i = 0; i < n; i++)
        ctxB->diag[i] = (i < n / 2) ? 1.0 : -1.0;

    LinearOperator_d_t A = { .rows = n, .cols = n, .matvec = diag_matvec_d,
                             .ctx = (linop_ctx_t *)ctxA };
    LinearOperator_d_t B = { .rows = n, .cols = n, .matvec = diag_matvec_d,
                             .ctx = (linop_ctx_t *)ctxB };

    f64 *S    = xcalloc(n * sizeSub, sizeof(f64));
    for (uint64_t j = 0; j < sizeSub; j++) S[j + j * n] = 1.0;

    const uint64_t n_rem = (mult - 1) * nev;
    f64 *Cx   = xcalloc(sizeSub * nev, sizeof(f64));
    f64 *Cp   = xcalloc(sizeSub * n_rem, sizeof(f64));
    f64 *eigVal = xcalloc(nev, sizeof(f64));
    int8_t *sig = xcalloc(sizeSub, sizeof(int8_t));
    f64 *wrk1 = xcalloc(sizeSub * sizeSub, sizeof(f64));
    f64 *wrk2 = xcalloc(n * sizeSub, sizeof(f64));
    f64 *wrk3 = xcalloc(sizeSub * sizeSub, sizeof(f64));
    f64 *wrk4 = xcalloc(sizeSub * sizeSub, sizeof(f64));

    d_indefinite_rayleigh_ritz_modified(n, nev, mult, 0, 0,
                                         S, wrk1, wrk2, wrk3, wrk4,
                                         Cx, Cp, eigVal, sig, &A, &B);

    f64 expected_eig[] = {1.0, 2.0, 3.0};
    f64 max_err = 0;
    printf("  d_indefinite_rr_modified (mult=2):\n");
    for (uint64_t i = 0; i < nev; i++) {
        f64 err = fabs(eigVal[i] - expected_eig[i]);
        if (err > max_err) max_err = err;
        printf("    λ[%lu] = %8.4f (exp %4.1f, err=%.3e)\n",
               (unsigned long)i, eigVal[i], expected_eig[i], err);
    }

    /* Verify signature: first 3 positive, rest negative */
    int sig_ok = 1;
    for (uint64_t i = 0; i < nev; i++)
        if (sig[i] != 1) sig_ok = 0;
    for (uint64_t i = nev; i < sizeSub; i++)
        if (sig[i] != -1) sig_ok = 0;

    /* Verify Cx reconstruction: X_new = S * Cx */
    f64 *X_new = xcalloc(n * nev, sizeof(f64));
    d_gemm_nn(n, nev, sizeSub, 1.0, S, Cx, 0.0, X_new);

    f64 recon_err = 0;
    for (uint64_t j = 0; j < nev; j++) {
        f64 col_norm = 0;
        for (uint64_t i = 0; i < n; i++)
            col_norm += X_new[i + j * n] * X_new[i + j * n];
        col_norm = sqrt(col_norm);
        if (col_norm > 1e-14) {
            f64 dominant = fabs(X_new[j + j * n]) / col_norm;
            f64 err = fabs(dominant - 1.0);
            if (err > recon_err) recon_err = err;
        }
    }
    printf("    Cx recon_err = %.3e  sig_ok = %d\n", recon_err, sig_ok);

    int pass = (max_err < TEST_TOL_D) && sig_ok && (recon_err < TEST_TOL_D);
    printf("    %s\n", pass ? "PASS" : "FAIL");

    safe_free((void**)&S);    safe_free((void**)&Cx);   safe_free((void**)&Cp);
    safe_free((void**)&eigVal); safe_free((void**)&sig);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2);
    safe_free((void**)&wrk3); safe_free((void**)&wrk4);
    safe_free((void**)&X_new);
    safe_free((void**)&ctxA->diag); safe_free((void**)&ctxA);
    safe_free((void**)&ctxB->diag); safe_free((void**)&ctxB);
    return pass;
}

/* ================================================================
 * Test: indefinite_rayleigh_ritz_modified — double, mult=3
 * ================================================================ */
int test_indef_rr_modified_d_mult3(void) {
    const uint64_t n = 9, nev = 3;
    const uint64_t mult = 3;
    const uint64_t sizeSub = mult * nev;

    diag_ctx_d_t *ctxA = xcalloc(1, sizeof(diag_ctx_d_t));
    ctxA->n = n;
    ctxA->diag = xcalloc(n, sizeof(f64));
    for (uint64_t i = 0; i < n; i++) ctxA->diag[i] = (f64)(i + 1);

    diag_ctx_d_t *ctxB = xcalloc(1, sizeof(diag_ctx_d_t));
    ctxB->n = n;
    ctxB->diag = xcalloc(n, sizeof(f64));
    /* First half positive, second half negative */
    for (uint64_t i = 0; i < n; i++)
        ctxB->diag[i] = (i < 5) ? 1.0 : -1.0;

    LinearOperator_d_t A = { .rows = n, .cols = n, .matvec = diag_matvec_d,
                             .ctx = (linop_ctx_t *)ctxA };
    LinearOperator_d_t B = { .rows = n, .cols = n, .matvec = diag_matvec_d,
                             .ctx = (linop_ctx_t *)ctxB };

    f64 *S    = xcalloc(n * sizeSub, sizeof(f64));
    for (uint64_t j = 0; j < sizeSub; j++) S[j + j * n] = 1.0;

    const uint64_t n_rem = (mult - 1) * nev;
    f64 *Cx   = xcalloc(sizeSub * nev, sizeof(f64));
    f64 *Cp   = xcalloc(sizeSub * n_rem, sizeof(f64));
    f64 *eigVal = xcalloc(nev, sizeof(f64));
    int8_t *sig = xcalloc(sizeSub, sizeof(int8_t));
    f64 *wrk1 = xcalloc(sizeSub * sizeSub, sizeof(f64));
    f64 *wrk2 = xcalloc(n * sizeSub, sizeof(f64));
    f64 *wrk3 = xcalloc(sizeSub * sizeSub, sizeof(f64));
    f64 *wrk4 = xcalloc(sizeSub * sizeSub, sizeof(f64));

    d_indefinite_rayleigh_ritz_modified(n, nev, mult, 0, 0,
                                         S, wrk1, wrk2, wrk3, wrk4,
                                         Cx, Cp, eigVal, sig, &A, &B);

    /* First 3 sorted positive eigenvalues: 1, 2, 3 */
    f64 expected_eig[] = {1.0, 2.0, 3.0};
    f64 max_err = 0;
    for (uint64_t i = 0; i < nev; i++) {
        f64 err = fabs(eigVal[i] - expected_eig[i]);
        if (err > max_err) max_err = err;
    }

    int pass = (max_err < TEST_TOL_D);
    printf("  d_indefinite_rr_modified (mult=3): max_err = %.3e  %s\n",
           max_err, pass ? "PASS" : "FAIL");

    safe_free((void**)&S);    safe_free((void**)&Cx);   safe_free((void**)&Cp);
    safe_free((void**)&eigVal); safe_free((void**)&sig);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2);
    safe_free((void**)&wrk3); safe_free((void**)&wrk4);
    safe_free((void**)&ctxA->diag); safe_free((void**)&ctxA);
    safe_free((void**)&ctxB->diag); safe_free((void**)&ctxB);
    return pass;
}

/* ================================================================
 * Test: indefinite_rayleigh_ritz_modified — complex double, mult=2
 * ================================================================ */
int test_indef_rr_modified_z(void) {
    const uint64_t n = 6, nev = 3;
    const uint64_t mult = 2;
    const uint64_t sizeSub = mult * nev;

    diag_ctx_z_t *ctxA = xcalloc(1, sizeof(diag_ctx_z_t));
    ctxA->n = n;
    ctxA->diag = xcalloc(n, sizeof(f64));
    for (uint64_t i = 0; i < n; i++) ctxA->diag[i] = (f64)(i + 1);

    diag_ctx_z_t *ctxB = xcalloc(1, sizeof(diag_ctx_z_t));
    ctxB->n = n;
    ctxB->diag = xcalloc(n, sizeof(f64));
    for (uint64_t i = 0; i < n; i++)
        ctxB->diag[i] = (i < n / 2) ? 1.0 : -1.0;

    LinearOperator_z_t A = { .rows = n, .cols = n, .matvec = diag_matvec_z,
                             .ctx = (linop_ctx_t *)ctxA };
    LinearOperator_z_t B = { .rows = n, .cols = n, .matvec = diag_matvec_z,
                             .ctx = (linop_ctx_t *)ctxB };

    c64 *S    = xcalloc(n * sizeSub, sizeof(c64));
    for (uint64_t j = 0; j < sizeSub; j++) S[j + j * n] = 1.0 + 0 * I;

    const uint64_t n_rem = (mult - 1) * nev;
    c64 *Cx   = xcalloc(sizeSub * nev, sizeof(c64));
    c64 *Cp   = xcalloc(sizeSub * n_rem, sizeof(c64));
    f64 *eigVal = xcalloc(nev, sizeof(f64));
    int8_t *sig = xcalloc(sizeSub, sizeof(int8_t));
    c64 *wrk1 = xcalloc(sizeSub * sizeSub, sizeof(c64));
    c64 *wrk2 = xcalloc(n * sizeSub, sizeof(c64));
    c64 *wrk3 = xcalloc(sizeSub * sizeSub, sizeof(c64));
    c64 *wrk4 = xcalloc(sizeSub * sizeSub, sizeof(c64));

    z_indefinite_rayleigh_ritz_modified(n, nev, mult, 0, 0,
                                         S, wrk1, wrk2, wrk3, wrk4,
                                         Cx, Cp, eigVal, sig, &A, &B);

    f64 expected_eig[] = {1.0, 2.0, 3.0};
    f64 max_err = 0;
    for (uint64_t i = 0; i < nev; i++) {
        f64 err = fabs(eigVal[i] - expected_eig[i]);
        if (err > max_err) max_err = err;
    }

    int pass = (max_err < TEST_TOL_D);
    printf("  z_indefinite_rr_modified (mult=2): max_err = %.3e  %s\n",
           max_err, pass ? "PASS" : "FAIL");

    safe_free((void**)&S);    safe_free((void**)&Cx);   safe_free((void**)&Cp);
    safe_free((void**)&eigVal); safe_free((void**)&sig);
    safe_free((void**)&wrk1); safe_free((void**)&wrk2);
    safe_free((void**)&wrk3); safe_free((void**)&wrk4);
    safe_free((void**)&ctxA->diag); safe_free((void**)&ctxA);
    safe_free((void**)&ctxB->diag); safe_free((void**)&ctxB);
    return pass;
}

/* ================================================================ */
int main(void) {
    int pass = 1;
    int result;

    printf("=== Indefinite Rayleigh-Ritz ===\n");

    result = test_indef_rr_d();  pass &= result;
    result = test_indef_rr_s();  pass &= result;
    result = test_indef_rr_z();  pass &= result;
    result = test_indef_rr_c();  pass &= result;

    printf("\n=== Modified Indefinite Rayleigh-Ritz ===\n");

    result = test_indef_rr_modified_d();       pass &= result;
    result = test_indef_rr_modified_d_mult3(); pass &= result;
    result = test_indef_rr_modified_z();       pass &= result;

    printf("\n%s\n", pass ? "All tests PASSED" : "Some tests FAILED");
    return pass ? 0 : 1;
}
