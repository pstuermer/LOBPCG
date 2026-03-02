/**
 * @file test_ortho_indefinite_mat.c
 * @brief Test matrix-based indefinite B-orthogonalization of U against V
 *
 * Tests ortho_indefinite_mat with indefinite diagonal matrix.
 * Verifies: ||V^H*mat*U||_F < tol and ||U^H*mat*U - I_sig||_F < tol
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
#include "lobpcg/memory.h"

#define TOL_F64 1e-10

#include "test_macros.h"

/* Create indefinite diagonal matrix: first n_pos entries = +1, rest = -1 */
static void make_indef_diag_d(uint64_t m, uint64_t n_pos, f64 *mat) {
  memset(mat, 0, m * m * sizeof(f64));
  for (uint64_t i = 0; i < m; i++)
    mat[i + i*m] = (i < n_pos) ? 1.0 : -1.0;
}

static void make_indef_diag_z(uint64_t m, uint64_t n_pos, c64 *mat) {
  memset(mat, 0, m * m * sizeof(c64));
  for (uint64_t i = 0; i < m; i++)
    mat[i + i*m] = (i < n_pos) ? 1.0 : -1.0;
}

/* ====================================================================
 * Tests
 * ==================================================================== */

TEST(d_ortho_indefinite_mat) {
  const uint64_t m = 100;
  const uint64_t n_u = 5, n_v = 5;
  const uint64_t n_pos = 60;
  const f64 eps_ortho = 1e-14;
  const f64 eps_drop = 1e-14;
  const uint64_t max_n = n_u > n_v ? n_u : n_v;

  f64 *U    = xcalloc(m * n_u, sizeof(f64));
  f64 *V    = xcalloc(m * n_v, sizeof(f64));
  f64 *mat  = xcalloc(m * m, sizeof(f64));
  f64 *wrk1 = xcalloc(m * max_n, sizeof(f64));
  f64 *wrk2 = xcalloc(m * max_n, sizeof(f64));
  f64 *wrk3 = xcalloc(m * max_n, sizeof(f64));

  make_indef_diag_d(m, n_pos, mat);
  for (uint64_t i = 0; i < m * n_u; i++) U[i] = (f64)rand()/RAND_MAX - 0.5;
  for (uint64_t i = 0; i < m * n_v; i++) V[i] = (f64)rand()/RAND_MAX - 0.5;

  f64 *tmp  = xcalloc(m * n_u, sizeof(f64));
  f64 *VtMU = xcalloc(n_v * n_u, sizeof(f64));
  f64 *UtMU = xcalloc(n_u * n_u, sizeof(f64));

  d_svqb_mat(m, n_v, eps_drop, 'n', V, mat, wrk1, wrk2, wrk3);

  d_gemm_nn(m, n_u, m, 1.0, mat, U, 0.0, tmp);
  d_gemm_tn(n_v, n_u, m, 1.0, V, tmp, 0.0, VtMU);
  f64 cross_pre = d_nrm2(n_v * n_u, VtMU);
  d_gemm_tn(n_u, n_u, m, 1.0, U, tmp, 0.0, UtMU);
  for (uint64_t i = 0; i < n_u; i++)
    UtMU[i + i*n_u] = fabs(UtMU[i + i*n_u]) - 1.0;
  f64 norm_pre = d_nrm2(n_u * n_u, UtMU);

  d_ortho_indefinite_mat(m, n_u, n_v, eps_ortho, eps_drop,
                         U, V, mat, wrk1, wrk2, wrk3);

  d_gemm_nn(m, n_u, m, 1.0, mat, U, 0.0, tmp);
  d_gemm_tn(n_v, n_u, m, 1.0, V, tmp, 0.0, VtMU);
  f64 cross_err = d_nrm2(n_v * n_u, VtMU);

  d_gemm_tn(n_u, n_u, m, 1.0, U, tmp, 0.0, UtMU);
  for (uint64_t i = 0; i < n_u; i++)
    UtMU[i + i*n_u] = fabs(UtMU[i + i*n_u]) - 1.0;
  f64 self_err = d_nrm2(n_u * n_u, UtMU);

  printf("pre: cross=%.2e norm=%.2e  post: cross=%.2e norm=%.2e ", cross_pre, norm_pre, cross_err, self_err);
  ASSERT(cross_err < TOL_F64);
  ASSERT(self_err < TOL_F64);

  safe_free((void**)&U); safe_free((void**)&V); safe_free((void**)&mat);
  safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3);
  safe_free((void**)&tmp); safe_free((void**)&VtMU); safe_free((void**)&UtMU);
}

TEST(z_ortho_indefinite_mat) {
  const uint64_t m = 100;
  const uint64_t n_u = 5, n_v = 5;
  const uint64_t n_pos = 60;
  const f64 eps_ortho = 1e-14;
  const f64 eps_drop = 1e-14;
  const uint64_t max_n = n_u > n_v ? n_u : n_v;

  c64 *U    = xcalloc(m * n_u, sizeof(c64));
  c64 *V    = xcalloc(m * n_v, sizeof(c64));
  c64 *mat  = xcalloc(m * m, sizeof(c64));
  c64 *wrk1 = xcalloc(m * max_n, sizeof(c64));
  c64 *wrk2 = xcalloc(m * max_n, sizeof(c64));
  c64 *wrk3 = xcalloc(m * max_n, sizeof(c64));

  make_indef_diag_z(m, n_pos, mat);
  for (uint64_t i = 0; i < m * n_u; i++)
    U[i] = ((f64)rand()/RAND_MAX - 0.5) + I*((f64)rand()/RAND_MAX - 0.5);
  for (uint64_t i = 0; i < m * n_v; i++)
    V[i] = ((f64)rand()/RAND_MAX - 0.5) + I*((f64)rand()/RAND_MAX - 0.5);

  c64 *tmp  = xcalloc(m * n_u, sizeof(c64));
  c64 *VhMU = xcalloc(n_v * n_u, sizeof(c64));
  c64 *UhMU = xcalloc(n_u * n_u, sizeof(c64));

  z_svqb_mat(m, n_v, eps_drop, 'n', V, mat, wrk1, wrk2, wrk3);

  z_gemm_nn(m, n_u, m, 1.0, mat, U, 0.0, tmp);
  z_gemm_hn(n_v, n_u, m, 1.0, V, tmp, 0.0, VhMU);
  f64 cross_pre = z_nrm2(n_v * n_u, VhMU);
  z_gemm_hn(n_u, n_u, m, 1.0, U, tmp, 0.0, UhMU);
  for (uint64_t i = 0; i < n_u; i++)
    UhMU[i + i*n_u] = fabs(creal(UhMU[i + i*n_u])) - 1.0;
  f64 norm_pre = z_nrm2(n_u * n_u, UhMU);

  z_ortho_indefinite_mat(m, n_u, n_v, eps_ortho, eps_drop,
                         U, V, mat, wrk1, wrk2, wrk3);

  z_gemm_nn(m, n_u, m, 1.0, mat, U, 0.0, tmp);
  z_gemm_hn(n_v, n_u, m, 1.0, V, tmp, 0.0, VhMU);
  f64 cross_err = z_nrm2(n_v * n_u, VhMU);

  z_gemm_hn(n_u, n_u, m, 1.0, U, tmp, 0.0, UhMU);
  for (uint64_t i = 0; i < n_u; i++)
    UhMU[i + i*n_u] = fabs(creal(UhMU[i + i*n_u])) - 1.0;
  f64 self_err = z_nrm2(n_u * n_u, UhMU);

  printf("pre: cross=%.2e norm=%.2e  post: cross=%.2e norm=%.2e ", cross_pre, norm_pre, cross_err, self_err);
  ASSERT(cross_err < TOL_F64);
  ASSERT(self_err < TOL_F64);

  safe_free((void**)&U); safe_free((void**)&V); safe_free((void**)&mat);
  safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3);
  safe_free((void**)&tmp); safe_free((void**)&VhMU); safe_free((void**)&UhMU);
}

int main(void) {
  srand((unsigned)time(NULL));

  printf("ortho_indefinite_mat tests:\n");
  RUN(d_ortho_indefinite_mat);
  RUN(z_ortho_indefinite_mat);

  printf("\n========================================\n");
  printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
  printf("========================================\n");
  return tests_failed > 0 ? 1 : 0;
}
