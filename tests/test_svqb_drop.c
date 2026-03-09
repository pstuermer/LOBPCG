/**
 * @file test_svqb_drop.c
 * @brief Tests for SVQB column dropping of linearly dependent vectors
 *
 * Constructs matrices with deliberate linear dependence and verifies
 * that svqb with drop='y' removes the dependent columns.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <complex.h>
#include "lobpcg/types.h"
#include "lobpcg.h"
#include "lobpcg/blas_wrapper.h"
#include "lobpcg/linop.h"
#include "lobpcg/memory.h"

#define TOL_F64 1e-12
#define TAU_F64 1e-12

#include "test_macros.h"

/* ====================================================================
 * Helpers
 * ==================================================================== */

static void fill_random_d(uint64_t n, f64 *x) {
  for (uint64_t i = 0; i < n; i++)
    x[i] = (f64)rand() / RAND_MAX - 0.5;
}

static void fill_random_z(uint64_t n, c64 *x) {
  for (uint64_t i = 0; i < n; i++)
    x[i] = ((f64)rand()/RAND_MAX - 0.5) + I*((f64)rand()/RAND_MAX - 0.5);
}

/* Compute ||U^H*U - I||_F for the first n columns of U (m x n) */
static f64 ortho_error_d(uint64_t m, uint64_t n, const f64 *U) {
  f64 *G = xcalloc(n * n, sizeof(f64));
  d_syrk(m, n, 1.0, U, 0.0, G);

  for (uint64_t j = 0; j < n; j++) {
    for (uint64_t i = j; i < n; i++) {
      if (i == j) G[i + j*n] -= 1.0;
      else G[i + j*n] = G[j + i*n];
    }
  }

  f64 err = d_nrm2(n*n, G);
  safe_free((void**)&G);
  return err;
}

static f64 ortho_error_z(uint64_t m, uint64_t n, const c64 *U) {
  c64 *G = xcalloc(n * n, sizeof(c64));
  z_herk(m, n, 1.0, U, 0.0, G);
  for (uint64_t j = 0; j < n; j++) {
    for (uint64_t i = j; i < n; i++) {
      if (i == j) G[i + j*n] -= 1.0;
      else G[i + j*n] = conj(G[j + i*n]);
    }
  }
  f64 err = z_nrm2(n*n, G);
  safe_free((void**)&G);
  return err;
}

/* ====================================================================
 * Double precision dropping tests
 * ==================================================================== */

TEST(d_svqb_drop_exact_duplicate) {
  /* U has 5 columns but col4 = col0 -> rank 4, should drop 1 */
  const uint64_t m = 100, n = 5;
  f64 *U = xcalloc(m * n, sizeof(f64));
  f64 *wrk1 = xcalloc(n * n, sizeof(f64));
  f64 *wrk2 = xcalloc(m * n, sizeof(f64));
  f64 *wrk3 = xcalloc(m * n, sizeof(f64));

  fill_random_d(m * n, U);
  memcpy(&U[4*m], &U[0*m], m * sizeof(f64));

  const uint64_t ncols = d_svqb(m, n, TAU_F64, 'y', U, wrk1, wrk2, wrk3, NULL);
  printf("ncols=%lu (expected %lu) ", ncols, n - 1);
  ASSERT(ncols == n - 1);

  f64 err = ortho_error_d(m, ncols, U);
  printf("ortho_err=%.2e ", err);
  ASSERT(err < TOL_F64 * n);

  safe_free((void**)&U); safe_free((void**)&wrk1);
  safe_free((void**)&wrk2); safe_free((void**)&wrk3);
}

TEST(d_svqb_drop_linear_combination) {
  /* col3 = 0.5*col0 + 0.5*col1 -> rank 4, should drop 1 */
  const uint64_t m = 100, n = 5;
  f64 *U = xcalloc(m * n, sizeof(f64));
  f64 *wrk1 = xcalloc(n * n, sizeof(f64));
  f64 *wrk2 = xcalloc(m * n, sizeof(f64));
  f64 *wrk3 = xcalloc(m * n, sizeof(f64));

  fill_random_d(m * n, U);
  for (uint64_t i = 0; i < m; i++)
    U[i + 3*m] = 0.5 * U[i + 0*m] + 0.5 * U[i + 1*m];

  const uint64_t ncols = d_svqb(m, n, TAU_F64, 'y', U, wrk1, wrk2, wrk3, NULL);
  printf("ncols=%lu (expected %lu) ", ncols, n - 1);
  ASSERT(ncols == n - 1);

  f64 err = ortho_error_d(m, ncols, U);
  printf("ortho_err=%.2e ", err);
  ASSERT(err < TOL_F64 * n);

  safe_free((void**)&U); safe_free((void**)&wrk1);
  safe_free((void**)&wrk2); safe_free((void**)&wrk3);
}

TEST(d_svqb_drop_multiple) {
  /* 6 columns, but col3=col0, col5=col2 -> rank 4, should drop 2 */
  const uint64_t m = 100, n = 6;
  f64 *U = xcalloc(m * n, sizeof(f64));
  f64 *wrk1 = xcalloc(n * n, sizeof(f64));
  f64 *wrk2 = xcalloc(m * n, sizeof(f64));
  f64 *wrk3 = xcalloc(m * n, sizeof(f64));

  fill_random_d(m * n, U);
  memcpy(&U[3*m], &U[0*m], m * sizeof(f64));
  memcpy(&U[5*m], &U[2*m], m * sizeof(f64));

  const uint64_t ncols = d_svqb(m, n, TAU_F64, 'y', U, wrk1, wrk2, wrk3, NULL);
  printf("ncols=%lu (expected %lu) ", ncols, n - 2);
  ASSERT(ncols == n - 2);

  f64 err = ortho_error_d(m, ncols, U);
  printf("ortho_err=%.2e ", err);
  ASSERT(err < TOL_F64 * n);

  safe_free((void**)&U); safe_free((void**)&wrk1);
  safe_free((void**)&wrk2); safe_free((void**)&wrk3);
}

TEST(d_svqb_drop_zero_column) {
  /* col2 = 0 -> rank 4, should drop 1 */
  const uint64_t m = 100, n = 5;
  f64 *U = xcalloc(m * n, sizeof(f64));
  f64 *wrk1 = xcalloc(n * n, sizeof(f64));
  f64 *wrk2 = xcalloc(m * n, sizeof(f64));
  f64 *wrk3 = xcalloc(m * n, sizeof(f64));

  fill_random_d(m * n, U);
  memset(&U[2*m], 0, m * sizeof(f64));

  const uint64_t ncols = d_svqb(m, n, TAU_F64, 'y', U, wrk1, wrk2, wrk3, NULL);
  printf("ncols=%lu (expected %lu) ", ncols, n - 1);
  ASSERT(ncols == n - 1);

  f64 err = ortho_error_d(m, ncols, U);
  printf("ortho_err=%.2e ", err);
  ASSERT(err < TOL_F64 * n);

  safe_free((void**)&U); safe_free((void**)&wrk1);
  safe_free((void**)&wrk2); safe_free((void**)&wrk3);
}

TEST(d_svqb_drop_independent_keeps_all) {
  /* All columns independent -> should retain all */
  const uint64_t m = 100, n = 5;
  f64 *U = xcalloc(m * n, sizeof(f64));
  f64 *wrk1 = xcalloc(n * n, sizeof(f64));
  f64 *wrk2 = xcalloc(m * n, sizeof(f64));
  f64 *wrk3 = xcalloc(m * n, sizeof(f64));

  fill_random_d(m * n, U);

  const uint64_t ncols = d_svqb(m, n, TAU_F64, 'y', U, wrk1, wrk2, wrk3, NULL);
  printf("ncols=%lu (expected %lu) ", ncols, n);
  ASSERT(ncols == n);

  f64 err = ortho_error_d(m, ncols, U);
  printf("ortho_err=%.2e ", err);
  ASSERT(err < TOL_F64 * n);

  safe_free((void**)&U); safe_free((void**)&wrk1);
  safe_free((void**)&wrk2); safe_free((void**)&wrk3);
}

TEST(d_svqb_nodrop_when_n) {
  /* With drop='n', all columns retained even with duplicates */
  const uint64_t m = 100, n = 5;
  f64 *U = xcalloc(m * n, sizeof(f64));
  f64 *wrk1 = xcalloc(n * n, sizeof(f64));
  f64 *wrk2 = xcalloc(m * n, sizeof(f64));
  f64 *wrk3 = xcalloc(m * n, sizeof(f64));

  fill_random_d(m * n, U);
  memcpy(&U[4*m], &U[0*m], m * sizeof(f64));

  const uint64_t ncols = d_svqb(m, n, TAU_F64, 'n', U, wrk1, wrk2, wrk3, NULL);
  printf("ncols=%lu (expected %lu) ", ncols, n);
  ASSERT(ncols == n);

  safe_free((void**)&U); safe_free((void**)&wrk1);
  safe_free((void**)&wrk2); safe_free((void**)&wrk3);
}

/* ====================================================================
 * Double complex dropping tests
 * ==================================================================== */

TEST(z_svqb_drop_exact_duplicate) {
  const uint64_t m = 100, n = 5;
  c64 *U = xcalloc(m * n, sizeof(c64));
  c64 *wrk1 = xcalloc(n * n, sizeof(c64));
  c64 *wrk2 = xcalloc(m * n, sizeof(c64));
  c64 *wrk3 = xcalloc(m * n, sizeof(c64));

  fill_random_z(m * n, U);
  memcpy(&U[4*m], &U[0*m], m * sizeof(c64));

  const uint64_t ncols = z_svqb(m, n, TAU_F64, 'y', U, wrk1, wrk2, wrk3, NULL);
  printf("ncols=%lu (expected %lu) ", ncols, n - 1);
  ASSERT(ncols == n - 1);

  f64 err = ortho_error_z(m, ncols, U);
  printf("ortho_err=%.2e ", err);
  ASSERT(err < TOL_F64 * n);

  safe_free((void**)&U); safe_free((void**)&wrk1);
  safe_free((void**)&wrk2); safe_free((void**)&wrk3);
}

TEST(z_svqb_drop_independent_keeps_all) {
  const uint64_t m = 100, n = 5;
  c64 *U = xcalloc(m * n, sizeof(c64));
  c64 *wrk1 = xcalloc(n * n, sizeof(c64));
  c64 *wrk2 = xcalloc(m * n, sizeof(c64));
  c64 *wrk3 = xcalloc(m * n, sizeof(c64));

  fill_random_z(m * n, U);

  const uint64_t ncols = z_svqb(m, n, TAU_F64, 'y', U, wrk1, wrk2, wrk3, NULL);
  printf("ncols=%lu (expected %lu) ", ncols, n);
  ASSERT(ncols == n);

  f64 err = ortho_error_z(m, ncols, U);
  printf("ortho_err=%.2e ", err);
  ASSERT(err < TOL_F64 * n);

  safe_free((void**)&U); safe_free((void**)&wrk1);
  safe_free((void**)&wrk2); safe_free((void**)&wrk3);
}

/* ====================================================================
 * Main
 * ==================================================================== */

int main(void) {
  srand((unsigned)time(NULL));

  printf("SVQB dropping tests (double):\n");
  RUN(d_svqb_drop_exact_duplicate);
  RUN(d_svqb_drop_linear_combination);
  RUN(d_svqb_drop_multiple);
  RUN(d_svqb_drop_zero_column);
  RUN(d_svqb_drop_independent_keeps_all);
  RUN(d_svqb_nodrop_when_n);

  printf("\nSVQB dropping tests (double complex):\n");
  RUN(z_svqb_drop_exact_duplicate);
  RUN(z_svqb_drop_independent_keeps_all);

  printf("\n========================================\n");
  printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
  printf("========================================\n");

  return tests_failed > 0 ? 1 : 0;
}
