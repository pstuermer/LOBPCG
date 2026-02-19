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
#include "lobpcg/linop.h"
#include "lobpcg/blas_wrapper.h"
#include "lobpcg/memory.h"

#define TEST_TOLERANCE 1e-12

/* ================================================================
 * Helpers: diagonal operators (real and complex)
 * ================================================================ */

typedef struct {
  uint64_t n;
  f64 *diag;
} diag_ctx_d_t;

void diag_matvec_d(const LinearOperator_d_t *op, const f64 *x, f64 *y) {
  diag_ctx_d_t *ctx = (diag_ctx_d_t*)op->ctx;
  for (uint64_t i = 0; i < ctx->n; i++) {
    y[i] = ctx->diag[i] * x[i];
  }
}

void diag_cleanup_d(linop_ctx_t *ctx) {
  diag_ctx_d_t *dctx = (diag_ctx_d_t*)ctx;
  safe_free((void**)&dctx->diag);
  safe_free((void**)&dctx);
}

typedef struct {
  uint64_t n;
  c64 *diag;
} diag_ctx_z_t;

void diag_matvec_z(const LinearOperator_z_t *op, const c64 *x, c64 *y) {
  diag_ctx_z_t *ctx = (diag_ctx_z_t*)op->ctx;
  for (uint64_t i = 0; i < ctx->n; i++) {
    y[i] = ctx->diag[i] * x[i];
  }
}

void diag_cleanup_z(linop_ctx_t *ctx) {
  diag_ctx_z_t *dctx = (diag_ctx_z_t*)ctx;
  safe_free((void**)&dctx->diag);
  safe_free((void**)&dctx);
}

/* ================================================================
 * Helpers: dense 3x3 operators (real and complex)
 * ================================================================ */

typedef struct {
  uint64_t n;
  f64 *data;  /* column-major n x n */
} dense_ctx_d_t;

void dense_matvec_d(const LinearOperator_d_t *op, const f64 *x, f64 *y) {
  dense_ctx_d_t *ctx = (dense_ctx_d_t*)op->ctx;
  uint64_t n = ctx->n;
  for (uint64_t i = 0; i < n; i++) {
    y[i] = 0.0;
    for (uint64_t j = 0; j < n; j++) {
      y[i] += ctx->data[i + j*n] * x[j];
    }
  }
}

void dense_cleanup_d(linop_ctx_t *ctx) {
  dense_ctx_d_t *dctx = (dense_ctx_d_t*)ctx;
  safe_free((void**)&dctx->data);
  safe_free((void**)&dctx);
}

typedef struct {
  uint64_t n;
  c64 *data;  /* column-major n x n */
} dense_ctx_z_t;

void dense_matvec_z(const LinearOperator_z_t *op, const c64 *x, c64 *y) {
  dense_ctx_z_t *ctx = (dense_ctx_z_t*)op->ctx;
  uint64_t n = ctx->n;
  for (uint64_t i = 0; i < n; i++) {
    y[i] = 0.0;
    for (uint64_t j = 0; j < n; j++) {
      y[i] += ctx->data[i + j*n] * x[j];
    }
  }
}

void dense_cleanup_z(linop_ctx_t *ctx) {
  dense_ctx_z_t *dctx = (dense_ctx_z_t*)ctx;
  safe_free((void**)&dctx->data);
  safe_free((void**)&dctx);
}

/* ================================================================
 * Helper: create permutation matrix B = {{0,0,1},{0,1,0},{1,0,0}}
 * as dense operator (real and complex)
 * ================================================================ */

static void make_B_real(LinearOperator_d_t *B) {
  dense_ctx_d_t *ctx = xcalloc(1, sizeof(dense_ctx_d_t));
  ctx->n = 3;
  ctx->data = xcalloc(9, sizeof(f64));
  /* Column-major: B[0,2]=1, B[1,1]=1, B[2,0]=1 */
  ctx->data[2 + 0*3] = 1.0;  /* B[2,0] */
  ctx->data[1 + 1*3] = 1.0;  /* B[1,1] */
  ctx->data[0 + 2*3] = 1.0;  /* B[0,2] */
  B->rows = 3; B->cols = 3;
  B->matvec = dense_matvec_d;
  B->cleanup = dense_cleanup_d;
  B->ctx = ctx;
}

static void make_B_complex(LinearOperator_z_t *B) {
  dense_ctx_z_t *ctx = xcalloc(1, sizeof(dense_ctx_z_t));
  ctx->n = 3;
  ctx->data = xcalloc(9, sizeof(c64));
  ctx->data[2 + 0*3] = 1.0;
  ctx->data[1 + 1*3] = 1.0;
  ctx->data[0 + 2*3] = 1.0;
  B->rows = 3; B->cols = 3;
  B->matvec = dense_matvec_z;
  B->cleanup = dense_cleanup_z;
  B->ctx = ctx;
}

/* ================================================================
 * Test 1: get_residual with exact eigenvectors (real, existing)
 * ================================================================ */
int test_get_residual(void) {
  const uint64_t n = 10;
  const uint64_t nev = 3;

  diag_ctx_d_t *ctx = xcalloc(1, sizeof(diag_ctx_d_t));
  ctx->n = n;
  ctx->diag = xcalloc(n, sizeof(f64));
  for (uint64_t i = 0; i < n; i++) {
    ctx->diag[i] = (f64)(i + 1);
  }

  LinearOperator_d_t A;
  A.rows = n; A.cols = n;
  A.matvec = diag_matvec_d;
  A.cleanup = diag_cleanup_d;
  A.ctx = ctx;

  f64 *X = xcalloc(n * nev, sizeof(f64));
  f64 *eigVal = xcalloc(nev, sizeof(f64));
  for (uint64_t i = 0; i < nev; i++) {
    X[i + i*n] = 1.0;
    eigVal[i] = (f64)(i + 1);
  }

  f64 *R = xcalloc(n * nev, sizeof(f64));
  f64 *wrk = xcalloc(n * nev, sizeof(f64));

  d_get_residual(n, nev, X, NULL, R, eigVal, wrk, &A, NULL);

  f64 R_norm = d_nrm2(n * nev, R);
  printf("  ||R||_F = %.3e (should be < %.3e)\n", R_norm, TEST_TOLERANCE);

  int pass = (R_norm < TEST_TOLERANCE);

  safe_free((void**)&X); safe_free((void**)&eigVal); safe_free((void**)&R); safe_free((void**)&wrk);
  diag_cleanup_d(ctx);

  return pass;
}

/* ================================================================
 * Test 2: get_residual_norm with small random residuals (existing)
 * ================================================================ */
int test_get_residual_norm(void) {
  const uint64_t n = 10;
  const uint64_t nev = 3;

  f64 *W = xcalloc(n * nev, sizeof(f64));
  for (uint64_t i = 0; i < n * nev; i++) {
    W[i] = 1e-8 * ((f64)rand() / RAND_MAX);
  }

  f64 *eigVals = xcalloc(nev, sizeof(f64));
  for (uint64_t i = 0; i < nev; i++) {
    eigVals[i] = (f64)(i + 1);
  }

  f64 *resNorm = xcalloc(nev, sizeof(f64));
  f64 *wrk1 = xcalloc(n * nev, sizeof(f64));
  f64 *wrk2 = xcalloc(n * nev, sizeof(f64));
  f64 *wrk3 = xcalloc(nev * nev, sizeof(f64));

  f64 ANorm = 10.0;
  f64 BNorm = 1.0;

  d_get_residual_norm(n, nev, W, eigVals, resNorm,
                      wrk1, wrk2, wrk3, ANorm, BNorm, NULL);

  printf("  Residual norms:\n");
  int pass = 1;
  for (uint64_t i = 0; i < nev; i++) {
    printf("    resNorm[%lu] = %.3e\n", (unsigned long)i, resNorm[i]);
    if (resNorm[i] > 1e-7) pass = 0;
  }

  safe_free((void**)&W); safe_free((void**)&eigVals); safe_free((void**)&resNorm);
  safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3);

  return pass;
}

/* ================================================================
 * Test 3: get_residual with exact eigenvectors, complex
 * A = diag(1..10) as z, X = e_1..e_3, eigVal = {1,2,3}
 * ================================================================ */
int test_get_residual_complex(void) {
  const uint64_t n = 10;
  const uint64_t nev = 3;

  diag_ctx_z_t *ctx = xcalloc(1, sizeof(diag_ctx_z_t));
  ctx->n = n;
  ctx->diag = xcalloc(n, sizeof(c64));
  for (uint64_t i = 0; i < n; i++) {
    ctx->diag[i] = (f64)(i + 1) + 0.0*I;
  }

  LinearOperator_z_t A;
  A.rows = n; A.cols = n;
  A.matvec = diag_matvec_z;
  A.cleanup = diag_cleanup_z;
  A.ctx = ctx;

  c64 *X = xcalloc(n * nev, sizeof(c64));
  f64 *eigVal = xcalloc(nev, sizeof(f64));
  for (uint64_t i = 0; i < nev; i++) {
    X[i + i*n] = 1.0 + 0.0*I;
    eigVal[i] = (f64)(i + 1);
  }

  c64 *R = xcalloc(n * nev, sizeof(c64));
  c64 *wrk = xcalloc(n * nev, sizeof(c64));

  z_get_residual(n, nev, X, NULL, R, eigVal, wrk, &A, NULL);

  f64 R_norm = z_nrm2(n * nev, R);
  printf("  ||R||_F = %.3e (should be < %.3e)\n", R_norm, TEST_TOLERANCE);

  int pass = (R_norm < TEST_TOLERANCE);

  safe_free((void**)&X); safe_free((void**)&eigVal); safe_free((void**)&R); safe_free((void**)&wrk);
  diag_cleanup_z(ctx);

  return pass;
}

/* ================================================================
 * Test 4: get_residual non-eigenvector, real, no B
 * A = diag(1,2,3), x = [1,2,3]^T, lambda=2
 * Expected R = A*x - lambda*x = [1,4,9] - [2,4,6] = [-1,0,3]
 * ================================================================ */
int test_get_residual_noneigvec_real(void) {
  const uint64_t n = 3;
  const uint64_t nev = 1;

  diag_ctx_d_t *ctx = xcalloc(1, sizeof(diag_ctx_d_t));
  ctx->n = n;
  ctx->diag = xcalloc(n, sizeof(f64));
  ctx->diag[0] = 1.0; ctx->diag[1] = 2.0; ctx->diag[2] = 3.0;

  LinearOperator_d_t A;
  A.rows = n; A.cols = n;
  A.matvec = diag_matvec_d;
  A.cleanup = diag_cleanup_d;
  A.ctx = ctx;

  f64 X[3] = {1.0, 2.0, 3.0};
  f64 eigVal[1] = {2.0};
  f64 R[3] = {0};
  f64 wrk[3] = {0};

  d_get_residual(n, nev, X, NULL, R, eigVal, wrk, &A, NULL);

  f64 expected[3] = {-1.0, 0.0, 3.0};
  f64 err = 0;
  for (uint64_t i = 0; i < n; i++) err += (R[i] - expected[i]) * (R[i] - expected[i]);
  err = sqrt(err);
  printf("  R = [%.1f, %.1f, %.1f], expected [-1, 0, 3], err = %.3e\n", R[0], R[1], R[2], err);

  diag_cleanup_d(ctx);
  return (err < TEST_TOLERANCE);
}

/* ================================================================
 * Test 5: get_residual non-eigenvector, complex, no B
 * A = diag(1,2,3), x = [1+3i, 2+2i, 3+i]^T, lambda=2
 * Expected R = [-1-3i, 0, 3+i]
 * ================================================================ */
int test_get_residual_noneigvec_complex(void) {
  const uint64_t n = 3;
  const uint64_t nev = 1;

  diag_ctx_z_t *ctx = xcalloc(1, sizeof(diag_ctx_z_t));
  ctx->n = n;
  ctx->diag = xcalloc(n, sizeof(c64));
  ctx->diag[0] = 1.0; ctx->diag[1] = 2.0; ctx->diag[2] = 3.0;

  LinearOperator_z_t A;
  A.rows = n; A.cols = n;
  A.matvec = diag_matvec_z;
  A.cleanup = diag_cleanup_z;
  A.ctx = ctx;

  c64 X[3] = {1.0+3.0*I, 2.0+2.0*I, 3.0+1.0*I};
  f64 eigVal[1] = {2.0};
  c64 R[3] = {0};
  c64 wrk[3] = {0};

  z_get_residual(n, nev, X, NULL, R, eigVal, wrk, &A, NULL);

  c64 expected[3] = {-1.0-3.0*I, 0.0+0.0*I, 3.0+1.0*I};
  f64 err = 0;
  for (uint64_t i = 0; i < n; i++) {
    c64 d = R[i] - expected[i];
    err += creal(d)*creal(d) + cimag(d)*cimag(d);
  }
  err = sqrt(err);
  printf("  R = [%.1f%+.1fi, %.1f%+.1fi, %.1f%+.1fi], err = %.3e\n",
         creal(R[0]), cimag(R[0]), creal(R[1]), cimag(R[1]), creal(R[2]), cimag(R[2]), err);

  diag_cleanup_z(ctx);
  return (err < TEST_TOLERANCE);
}

/* ================================================================
 * Test 6: get_residual non-eigenvector, real, with B
 * A = diag(1,2,3), B = {{0,0,1},{0,1,0},{1,0,0}},
 * x = [1,2,3]^T, lambda=2
 * R = A*x - lambda*B*x = [1,4,9] - 2*[3,2,1] = [-5,0,7]
 * ================================================================ */
int test_get_residual_noneigvec_real_B(void) {
  const uint64_t n = 3;
  const uint64_t nev = 1;

  diag_ctx_d_t *ctx = xcalloc(1, sizeof(diag_ctx_d_t));
  ctx->n = n;
  ctx->diag = xcalloc(n, sizeof(f64));
  ctx->diag[0] = 1.0; ctx->diag[1] = 2.0; ctx->diag[2] = 3.0;

  LinearOperator_d_t A;
  A.rows = n; A.cols = n;
  A.matvec = diag_matvec_d;
  A.cleanup = diag_cleanup_d;
  A.ctx = ctx;

  LinearOperator_d_t B;
  make_B_real(&B);

  f64 X[3] = {1.0, 2.0, 3.0};
  f64 eigVal[1] = {2.0};
  f64 R[3] = {0};
  f64 wrk[3] = {0};

  d_get_residual(n, nev, X, NULL, R, eigVal, wrk, &A, &B);

  f64 expected[3] = {-5.0, 0.0, 7.0};
  f64 err = 0;
  for (uint64_t i = 0; i < n; i++) err += (R[i] - expected[i]) * (R[i] - expected[i]);
  err = sqrt(err);
  printf("  R = [%.1f, %.1f, %.1f], expected [-5, 0, 7], err = %.3e\n", R[0], R[1], R[2], err);

  diag_cleanup_d(ctx);
  dense_cleanup_d(B.ctx);
  return (err < TEST_TOLERANCE);
}

/* ================================================================
 * Test 7: get_residual non-eigenvector, complex, with B
 * A = diag(1,2,3), B = perm, x = [1+3i, 2+2i, 3+i]^T, lambda=2
 * B*x = [3+i, 2+2i, 1+3i]
 * R = A*x - 2*B*x = [1+3i, 4+4i, 9+3i] - [6+2i, 4+4i, 2+6i] = [-5+i, 0, 7-3i]
 * ================================================================ */
int test_get_residual_noneigvec_complex_B(void) {
  const uint64_t n = 3;
  const uint64_t nev = 1;

  diag_ctx_z_t *ctx = xcalloc(1, sizeof(diag_ctx_z_t));
  ctx->n = n;
  ctx->diag = xcalloc(n, sizeof(c64));
  ctx->diag[0] = 1.0; ctx->diag[1] = 2.0; ctx->diag[2] = 3.0;

  LinearOperator_z_t A;
  A.rows = n; A.cols = n;
  A.matvec = diag_matvec_z;
  A.cleanup = diag_cleanup_z;
  A.ctx = ctx;

  LinearOperator_z_t B;
  make_B_complex(&B);

  c64 X[3] = {1.0+3.0*I, 2.0+2.0*I, 3.0+1.0*I};
  f64 eigVal[1] = {2.0};
  c64 R[3] = {0};
  c64 wrk[3] = {0};

  z_get_residual(n, nev, X, NULL, R, eigVal, wrk, &A, &B);

  c64 expected[3] = {-5.0+1.0*I, 0.0+0.0*I, 7.0-3.0*I};
  f64 err = 0;
  for (uint64_t i = 0; i < n; i++) {
    c64 d = R[i] - expected[i];
    err += creal(d)*creal(d) + cimag(d)*cimag(d);
  }
  err = sqrt(err);
  printf("  R = [%.1f%+.1fi, %.1f%+.1fi, %.1f%+.1fi], err = %.3e\n",
         creal(R[0]), cimag(R[0]), creal(R[1]), cimag(R[1]), creal(R[2]), cimag(R[2]), err);

  diag_cleanup_z(ctx);
  dense_cleanup_z(B.ctx);
  return (err < TEST_TOLERANCE);
}

/* ================================================================
 * Test 8: get_residual_norm non-eigenvector, real, no B
 * R = [-1, 0, 3], ||R||_2 = sqrt(10)
 * ANorm=3, BNorm=1, lambda=2 => denom = 3 + 2*1 = 5
 * expected resNorm = sqrt(10)/5
 * ================================================================ */
int test_get_residual_norm_noneigvec(void) {
  const uint64_t n = 3;
  const uint64_t nev = 1;

  diag_ctx_d_t *ctx = xcalloc(1, sizeof(diag_ctx_d_t));
  ctx->n = n;
  ctx->diag = xcalloc(n, sizeof(f64));
  ctx->diag[0] = 1.0; ctx->diag[1] = 2.0; ctx->diag[2] = 3.0;

  LinearOperator_d_t A;
  A.rows = n; A.cols = n;
  A.matvec = diag_matvec_d;
  A.cleanup = diag_cleanup_d;
  A.ctx = ctx;

  f64 X[3] = {1.0, 2.0, 3.0};
  f64 eigVal[1] = {2.0};
  f64 R[3] = {0};
  f64 wrk[3] = {0};

  d_get_residual(n, nev, X, NULL, R, eigVal, wrk, &A, NULL);

  /* Now compute residual norm */
  f64 resNorm[1] = {0};
  f64 wrk1[3] = {0}, wrk2[3] = {0}, wrk3[1] = {0};
  f64 ANorm = 3.0, BNorm = 1.0;

  d_get_residual_norm(n, nev, R, eigVal, resNorm, wrk1, wrk2, wrk3, ANorm, BNorm, NULL);

  f64 expected = sqrt(10.0) / 5.0;
  f64 err = fabs(resNorm[0] - expected);
  printf("  resNorm = %.6e, expected = %.6e, err = %.3e\n", resNorm[0], expected, err);

  diag_cleanup_d(ctx);
  return (err < TEST_TOLERANCE);
}

/* ================================================================
 * Test 9: get_residual_norm non-eigenvector, complex, no B
 * R = [-1-3i, 0, 3+i], ||R||_2 = sqrt(1+9+0+9+1) = sqrt(20) = 2sqrt(5)
 * denom = 3 + 2*1 = 5
 * expected resNorm = 2*sqrt(5)/5
 * ================================================================ */
int test_get_residual_norm_noneigvec_complex(void) {
  const uint64_t n = 3;
  const uint64_t nev = 1;

  diag_ctx_z_t *ctx = xcalloc(1, sizeof(diag_ctx_z_t));
  ctx->n = n;
  ctx->diag = xcalloc(n, sizeof(c64));
  ctx->diag[0] = 1.0; ctx->diag[1] = 2.0; ctx->diag[2] = 3.0;

  LinearOperator_z_t A;
  A.rows = n; A.cols = n;
  A.matvec = diag_matvec_z;
  A.cleanup = diag_cleanup_z;
  A.ctx = ctx;

  c64 X[3] = {1.0+3.0*I, 2.0+2.0*I, 3.0+1.0*I};
  f64 eigVal[1] = {2.0};
  c64 R[3] = {0};
  c64 wrk[3] = {0};

  z_get_residual(n, nev, X, NULL, R, eigVal, wrk, &A, NULL);

  f64 resNorm[1] = {0};
  c64 wrk1[3] = {0}, wrk2[3] = {0}, wrk3[1] = {0};
  f64 ANorm = 3.0, BNorm = 1.0;

  z_get_residual_norm(n, nev, R, eigVal, resNorm, wrk1, wrk2, wrk3, ANorm, BNorm, NULL);

  f64 expected = 2.0 * sqrt(5.0) / 5.0;
  f64 err = fabs(resNorm[0] - expected);
  printf("  resNorm = %.6e, expected = %.6e, err = %.3e\n", resNorm[0], expected, err);

  diag_cleanup_z(ctx);
  return (err < TEST_TOLERANCE);
}

/* ================================================================
 * Test 10: get_residual_norm non-eigenvector, real, with B
 * R = [-5, 0, 7]
 * B*R = [7, 0, -5]
 * R^H B R = -5*7 + 0 + 7*(-5) = -35 + 0 - 35 = -70
 * nom = sqrt(|R^H B R|) = sqrt(70)
 * denom = 3 + 2*1 = 5
 * expected resNorm = sqrt(70)/5
 * ================================================================ */
int test_get_residual_norm_noneigvec_real_B(void) {
  const uint64_t n = 3;
  const uint64_t nev = 1;

  diag_ctx_d_t *ctx = xcalloc(1, sizeof(diag_ctx_d_t));
  ctx->n = n;
  ctx->diag = xcalloc(n, sizeof(f64));
  ctx->diag[0] = 1.0; ctx->diag[1] = 2.0; ctx->diag[2] = 3.0;

  LinearOperator_d_t A;
  A.rows = n; A.cols = n;
  A.matvec = diag_matvec_d;
  A.cleanup = diag_cleanup_d;
  A.ctx = ctx;

  LinearOperator_d_t B;
  make_B_real(&B);

  f64 X[3] = {1.0, 2.0, 3.0};
  f64 eigVal[1] = {2.0};
  f64 R[3] = {0};
  f64 wrk[3] = {0};

  d_get_residual(n, nev, X, NULL, R, eigVal, wrk, &A, &B);

  f64 resNorm[1] = {0};
  f64 wrk1[3] = {0}, wrk2[3] = {0}, wrk3[1] = {0};
  f64 ANorm = 3.0, BNorm = 1.0;

  d_get_residual_norm(n, nev, R, eigVal, resNorm, wrk1, wrk2, wrk3, ANorm, BNorm, &B);

  f64 expected = sqrt(70.0) / 5.0;
  f64 err = fabs(resNorm[0] - expected);
  printf("  resNorm = %.6e, expected = %.6e, err = %.3e\n", resNorm[0], expected, err);

  diag_cleanup_d(ctx);
  dense_cleanup_d(B.ctx);
  return (err < TEST_TOLERANCE);
}

/* ================================================================
 * Test 11: get_residual_norm non-eigenvector, complex, with B
 * R = [-5+i, 0, 7-3i]
 * B*R = [7-3i, 0, -5+i]
 * R^H B R = conj(-5+i)*(7-3i) + 0 + conj(7-3i)*(-5+i)
 *         = (-5-i)(7-3i) + (7+3i)(-5+i)
 *         = (-35+15i-7i+3i^2) + (-35+7i-15i+3i^2)
 *         = (-35+8i-3) + (-35-8i-3)
 *         = -38+8i + -38-8i = -76
 * nom = sqrt(|-76|) = sqrt(76) = 2*sqrt(19)
 * denom = 5
 * expected = 2*sqrt(19)/5
 * ================================================================ */
int test_get_residual_norm_noneigvec_complex_B(void) {
  const uint64_t n = 3;
  const uint64_t nev = 1;

  diag_ctx_z_t *ctx = xcalloc(1, sizeof(diag_ctx_z_t));
  ctx->n = n;
  ctx->diag = xcalloc(n, sizeof(c64));
  ctx->diag[0] = 1.0; ctx->diag[1] = 2.0; ctx->diag[2] = 3.0;

  LinearOperator_z_t A;
  A.rows = n; A.cols = n;
  A.matvec = diag_matvec_z;
  A.cleanup = diag_cleanup_z;
  A.ctx = ctx;

  LinearOperator_z_t B;
  make_B_complex(&B);

  c64 X[3] = {1.0+3.0*I, 2.0+2.0*I, 3.0+1.0*I};
  f64 eigVal[1] = {2.0};
  c64 R[3] = {0};
  c64 wrk[3] = {0};

  z_get_residual(n, nev, X, NULL, R, eigVal, wrk, &A, &B);

  f64 resNorm[1] = {0};
  c64 wrk1[3] = {0}, wrk2[3] = {0}, wrk3[1] = {0};
  f64 ANorm = 3.0, BNorm = 1.0;

  z_get_residual_norm(n, nev, R, eigVal, resNorm, wrk1, wrk2, wrk3, ANorm, BNorm, &B);

  f64 expected = 2.0 * sqrt(19.0) / 5.0;
  f64 err = fabs(resNorm[0] - expected);
  printf("  resNorm = %.6e, expected = %.6e, err = %.3e\n", resNorm[0], expected, err);

  diag_cleanup_z(ctx);
  dense_cleanup_z(B.ctx);
  return (err < TEST_TOLERANCE);
}

/* ================================================================
 * Main
 * ================================================================ */
int main(void) {
  printf("Testing residual functions...\n");

  int n_tests = 11;
  int pass_count = 0;

  struct { const char *name; int (*func)(void); } tests[] = {
    {"get_residual exact eigvecs (real)",               test_get_residual},
    {"get_residual_norm small random (real)",            test_get_residual_norm},
    {"get_residual exact eigvecs (complex)",             test_get_residual_complex},
    {"get_residual non-eigvec (real, no B)",             test_get_residual_noneigvec_real},
    {"get_residual non-eigvec (complex, no B)",          test_get_residual_noneigvec_complex},
    {"get_residual non-eigvec (real, with B)",            test_get_residual_noneigvec_real_B},
    {"get_residual non-eigvec (complex, with B)",         test_get_residual_noneigvec_complex_B},
    {"get_residual_norm non-eigvec (real, no B)",         test_get_residual_norm_noneigvec},
    {"get_residual_norm non-eigvec (complex, no B)",      test_get_residual_norm_noneigvec_complex},
    {"get_residual_norm non-eigvec (real, with B)",        test_get_residual_norm_noneigvec_real_B},
    {"get_residual_norm non-eigvec (complex, with B)",     test_get_residual_norm_noneigvec_complex_B},
  };

  for (int i = 0; i < n_tests; i++) {
    printf("\nTest %d: %s\n", i+1, tests[i].name);
    int result = tests[i].func();
    printf("  Result: %s\n", result ? "PASS" : "FAIL");
    if (result) pass_count++;
  }

  printf("\n%d/%d tests passed\n", pass_count, n_tests);
  return (pass_count == n_tests) ? 0 : 1;
}
