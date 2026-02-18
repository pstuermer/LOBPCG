/**
 * @file test_ortho_randomized_mat.c
 * @brief Test matrix-based B-orthogonalization of U against V
 *
 * Tests ortho_randomized_mat with indefinite diagonal matrix.
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
 * Double precision test
 * ==================================================================== */
int test_ortho_rmat_d(void) {
    const uint64_t m = 100;
    const uint64_t n_u = 5, n_v = 5;
    const uint64_t n_pos = 60;
    const f64 eps_ortho = 1e-14;
    const f64 eps_drop = 1e-14;
    const uint64_t max_n = n_u > n_v ? n_u : n_v;

    f64 *U = xcalloc(m * n_u, sizeof(f64));
    f64 *V = xcalloc(m * n_v, sizeof(f64));
    f64 *mat = xcalloc(m * m, sizeof(f64));
    f64 *wrk1 = xcalloc(m * max_n, sizeof(f64));
    f64 *wrk2 = xcalloc(m * max_n, sizeof(f64));
    f64 *wrk3 = xcalloc(max_n * max_n, sizeof(f64));

    make_indef_diag_d(m, n_pos, mat);

    /* Fill U and V with random values */
    for (uint64_t i = 0; i < m * n_u; i++) U[i] = (f64)rand()/RAND_MAX - 0.5;
    for (uint64_t i = 0; i < m * n_v; i++) V[i] = (f64)rand()/RAND_MAX - 0.5;

    /* Pre-orthonormalize V via svqb_mat */
    d_svqb_mat(m, n_v, eps_drop, 'n', V, mat, wrk1, wrk2, wrk3);

    /* Orthogonalize U against V */
    uint64_t n_ret = d_ortho_randomized_mat(m, n_u, n_v, eps_ortho, eps_drop,
                                             U, V, mat, wrk1, wrk2, wrk3);
    printf("  Returned %lu columns (expected %lu)\n",
           (unsigned long)n_ret, (unsigned long)n_u);

    /* Check ||V^T * mat * U||_F < tol */
    f64 *tmp = xcalloc(m * n_u, sizeof(f64));
    f64 *VtMU = xcalloc(n_v * n_u, sizeof(f64));
    d_gemm_nn(m, n_u, m, 1.0, mat, U, 0.0, tmp);
    d_gemm_tn(n_v, n_u, m, 1.0, V, tmp, 0.0, VtMU);
    f64 cross_err = d_nrm2(n_v * n_u, VtMU);
    printf("  ||V^T*mat*U||_F = %.3e (tol = %.3e)\n", cross_err, TOL_F64);

    /* Check ||U^T * mat * U - I_sig||_F < tol */
    f64 *UtMU = xcalloc(n_u * n_u, sizeof(f64));
    d_gemm_tn(n_u, n_u, m, 1.0, U, tmp, 0.0, UtMU);
    for (uint64_t i = 0; i < n_u; i++)
        UtMU[i + i*n_u] = fabs(UtMU[i + i*n_u]) - 1.0;
    f64 self_err = d_nrm2(n_u * n_u, UtMU);
    printf("  ||U^T*mat*U - I_sig||_F = %.3e (tol = %.3e)\n", self_err, TOL_F64);

    int pass = (cross_err < TOL_F64) && (self_err < TOL_F64);
    safe_free((void**)&U); safe_free((void**)&V); safe_free((void**)&mat); safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3);
    safe_free((void**)&tmp); safe_free((void**)&VtMU); safe_free((void**)&UtMU);
    return pass;
}

/* ====================================================================
 * Complex double precision test
 * ==================================================================== */
int test_ortho_rmat_z(void) {
    const uint64_t m = 100;
    const uint64_t n_u = 5, n_v = 5;
    const uint64_t n_pos = 60;
    const f64 eps_ortho = 1e-14;
    const f64 eps_drop = 1e-14;
    const uint64_t max_n = n_u > n_v ? n_u : n_v;

    c64 *U = xcalloc(m * n_u, sizeof(c64));
    c64 *V = xcalloc(m * n_v, sizeof(c64));
    c64 *mat = xcalloc(m * m, sizeof(c64));
    c64 *wrk1 = xcalloc(m * max_n, sizeof(c64));
    c64 *wrk2 = xcalloc(m * max_n, sizeof(c64));
    c64 *wrk3 = xcalloc(max_n * max_n, sizeof(c64));

    make_indef_diag_z(m, n_pos, mat);

    for (uint64_t i = 0; i < m * n_u; i++)
        U[i] = ((f64)rand()/RAND_MAX - 0.5) + I*((f64)rand()/RAND_MAX - 0.5);
    for (uint64_t i = 0; i < m * n_v; i++)
        V[i] = ((f64)rand()/RAND_MAX - 0.5) + I*((f64)rand()/RAND_MAX - 0.5);

    /* Pre-orthonormalize V via svqb_mat */
    z_svqb_mat(m, n_v, eps_drop, 'n', V, mat, wrk1, wrk2, wrk3);

    /* Orthogonalize U against V */
    uint64_t n_ret = z_ortho_randomized_mat(m, n_u, n_v, eps_ortho, eps_drop,
                                             U, V, mat, wrk1, wrk2, wrk3);
    printf("  Returned %lu columns (expected %lu)\n",
           (unsigned long)n_ret, (unsigned long)n_u);

    /* Check ||V^H * mat * U||_F < tol */
    c64 *tmp = xcalloc(m * n_u, sizeof(c64));
    c64 *VhMU = xcalloc(n_v * n_u, sizeof(c64));
    z_gemm_nn(m, n_u, m, 1.0, mat, U, 0.0, tmp);
    z_gemm_hn(n_v, n_u, m, 1.0, V, tmp, 0.0, VhMU);
    f64 cross_err = z_nrm2(n_v * n_u, VhMU);
    printf("  ||V^H*mat*U||_F = %.3e (tol = %.3e)\n", cross_err, TOL_F64);

    /* Check ||U^H * mat * U - I_sig||_F < tol */
    c64 *UhMU = xcalloc(n_u * n_u, sizeof(c64));
    z_gemm_hn(n_u, n_u, m, 1.0, U, tmp, 0.0, UhMU);
    for (uint64_t i = 0; i < n_u; i++)
        UhMU[i + i*n_u] = fabs(creal(UhMU[i + i*n_u])) - 1.0;
    f64 self_err = z_nrm2(n_u * n_u, UhMU);
    printf("  ||U^H*mat*U - I_sig||_F = %.3e (tol = %.3e)\n", self_err, TOL_F64);

    int pass = (cross_err < TOL_F64) && (self_err < TOL_F64);
    safe_free((void**)&U); safe_free((void**)&V); safe_free((void**)&mat); safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3);
    safe_free((void**)&tmp); safe_free((void**)&VhMU); safe_free((void**)&UhMU);
    return pass;
}

int main(void) {
    srand((unsigned)time(NULL));

    printf("Testing ortho_randomized_mat...\n");

    printf("\nTest 1: ortho_randomized_mat_d (double real)\n");
    int t1 = test_ortho_rmat_d();
    printf("  Result: %s\n", t1 ? "PASS" : "FAIL");

    printf("\nTest 2: ortho_randomized_mat_z (complex double)\n");
    int t2 = test_ortho_rmat_z();
    printf("  Result: %s\n", t2 ? "PASS" : "FAIL");

    if (t1 && t2) {
        printf("\nAll tests PASSED\n");
        return 0;
    } else {
        printf("\nSome tests FAILED\n");
        return 1;
    }
}
