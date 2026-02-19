/**
 * @file test_estimate_norm.c
 * @brief Test estimate_norm (power iteration spectral radius estimation)
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

/* Dense 3x3 operator (real) */
typedef struct {
  uint64_t n;
  f64 *data;
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

/* Dense 3x3 operator (complex) */
typedef struct {
  uint64_t n;
  c64 *data;
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

/* ================================================================
 * Test 1: Real 3x3 matrix, entries 1..9
 * Column-major: [[1,4,7],[2,5,8],[3,6,9]]
 * Exact spectral radius = (15 + 3*sqrt(33))/2 ~ 16.117
 * ================================================================ */
int test_estimate_norm_real(void) {
  const uint64_t n = 3;

  dense_ctx_d_t *ctx = xcalloc(1, sizeof(dense_ctx_d_t));
  ctx->n = n;
  ctx->data = xcalloc(9, sizeof(f64));
  /* Column-major: data[i + j*3], fill with 1..9 running index */
  for (uint64_t k = 0; k < 9; k++) {
    ctx->data[k] = (f64)(k + 1);
  }
  /* Matrix is: [[1,4,7],[2,5,8],[3,6,9]] */

  LinearOperator_d_t A;
  A.rows = n; A.cols = n;
  A.matvec = dense_matvec_d;
  A.cleanup = NULL;
  A.ctx = ctx;

  f64 *wrk1 = xcalloc(n, sizeof(f64));
  f64 *wrk2 = xcalloc(n, sizeof(f64));

  f64 est = d_estimate_norm(n, &A, wrk1, wrk2);

  f64 exact = (15.0 + 3.0 * sqrt(33.0)) / 2.0;
  f64 rel_err = fabs(est - exact) / exact;
  printf("  estimate = %.6f, exact = %.6f, rel_err = %.3e\n", est, exact, rel_err);

  safe_free((void**)&wrk1); safe_free((void**)&wrk2);
  safe_free((void**)&ctx->data); safe_free((void**)&ctx);

  return (rel_err < 0.05);
}

/* ================================================================
 * Test 2: Complex 3x3 matrix
 * Entry k (0-based) = (k+1) + (9-k)i
 * Column-major: [[1+9i, 4+6i, 7+3i], [2+8i, 5+5i, 8+2i], [3+7i, 6+4i, 9+i]]
 * Compute reference spectral radius via z_geev
 * ================================================================ */
int test_estimate_norm_complex(void) {
  const uint64_t n = 3;

  dense_ctx_z_t *ctx = xcalloc(1, sizeof(dense_ctx_z_t));
  ctx->n = n;
  ctx->data = xcalloc(9, sizeof(c64));
  for (uint64_t k = 0; k < 9; k++) {
    ctx->data[k] = (f64)(k + 1) + (f64)(9 - k) * I;
  }

  LinearOperator_z_t A;
  A.rows = n; A.cols = n;
  A.matvec = dense_matvec_z;
  A.cleanup = NULL;
  A.ctx = ctx;

  /* Compute reference spectral radius via LAPACK z_geev */
  c64 *Acopy = xcalloc(9, sizeof(c64));
  for (uint64_t k = 0; k < 9; k++) Acopy[k] = ctx->data[k];

  c64 *w = xcalloc(n, sizeof(c64));
  c64 *VR = xcalloc(9, sizeof(c64));
  z_geev(n, Acopy, w, VR);

  f64 spectral_radius = 0.0;
  for (uint64_t i = 0; i < n; i++) {
    f64 mag = cabs(w[i]);
    printf("  eigenvalue[%lu] = %.6f + %.6fi  (|.|=%.6f)\n",
           (unsigned long)i, creal(w[i]), cimag(w[i]), mag);
    if (mag > spectral_radius) spectral_radius = mag;
  }

  c64 *wrk1 = xcalloc(n, sizeof(c64));
  c64 *wrk2 = xcalloc(n, sizeof(c64));

  f64 est = z_estimate_norm(n, &A, wrk1, wrk2);

  f64 rel_err = fabs(est - spectral_radius) / spectral_radius;
  printf("  estimate = %.6f, spectral_radius = %.6f, rel_err = %.3e\n",
         est, spectral_radius, rel_err);

  safe_free((void**)&wrk1); safe_free((void**)&wrk2);
  safe_free((void**)&Acopy); safe_free((void**)&w); safe_free((void**)&VR);
  safe_free((void**)&ctx->data); safe_free((void**)&ctx);

  return (rel_err < 0.05);
}

int main(void) {
  printf("Testing estimate_norm...\n");

  printf("\nTest 1: Real 3x3 (entries 1..9)\n");
  int t1 = test_estimate_norm_real();
  printf("  Result: %s\n", t1 ? "PASS" : "FAIL");

  printf("\nTest 2: Complex 3x3 (entries (k+1)+(9-k)i)\n");
  int t2 = test_estimate_norm_complex();
  printf("  Result: %s\n", t2 ? "PASS" : "FAIL");

  printf("\n%d/2 tests passed\n", t1 + t2);
  return (t1 && t2) ? 0 : 1;
}
