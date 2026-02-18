/**
 * @file test_svqb_mat.c
 * @brief Unit tests for matrix-based SVQB orthogonalization
 *
 * Tests svqb_mat with an indefinite diagonal matrix:
 *   mat = diag(1, ..., 1, -1, ..., -1)
 * After svqb_mat, we verify ||U^H*mat*U - I||_F < tol
 * where I has ±1 on the diagonal (the signs from the metric).
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

#define TOL_F64 1e-12

/**
 * Compute ||U^H*mat*U - I_sig||_F where I_sig has |diag| = 1
 * For indefinite metric, the Gram matrix U^H*mat*U should satisfy
 * diag entries having |value| ≈ 1 and off-diag ≈ 0.
 */
static f64 ortho_error_mat_d(uint64_t m, uint64_t n, const f64 *U,
                              const f64 *mat) {
    f64 *tmp = xcalloc(m * n, sizeof(f64));
    f64 *G = xcalloc(n * n, sizeof(f64));

    /* tmp = mat * U */
    d_gemm_nn(m, n, m, 1.0, mat, U, 0.0, tmp);
    /* G = U^T * tmp */
    d_gemm_tn(n, n, m, 1.0, U, tmp, 0.0, G);

    /* Subtract sign(G_ii) from diagonal */
    for (uint64_t i = 0; i < n; i++) {
        f64 sign = (G[i + i*n] >= 0) ? 1.0 : -1.0;
        G[i + i*n] = fabs(G[i + i*n]) - 1.0;
        (void)sign;
    }

    f64 err = d_nrm2(n * n, G);
    safe_free((void**)&tmp); safe_free((void**)&G);
    return err;
}

static f64 ortho_error_mat_z(uint64_t m, uint64_t n, const c64 *U,
                              const c64 *mat) {
    c64 *tmp = xcalloc(m * n, sizeof(c64));
    c64 *G = xcalloc(n * n, sizeof(c64));

    /* tmp = mat * U */
    z_gemm_nn(m, n, m, 1.0, mat, U, 0.0, tmp);
    /* G = U^H * tmp */
    z_gemm_hn(n, n, m, 1.0, U, tmp, 0.0, G);

    /* Subtract sign from diagonal: |Re(G_ii)| - 1 */
    for (uint64_t i = 0; i < n; i++)
        G[i + i*n] = fabs(creal(G[i + i*n])) - 1.0;

    f64 err = z_nrm2(n * n, G);
    safe_free((void**)&tmp); safe_free((void**)&G);
    return err;
}

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

int test_svqb_mat_d(void) {
    const uint64_t m = 100, n = 10;
    const uint64_t n_pos = 60;  /* 60 positive, 40 negative */

    f64 *U = xcalloc(m * n, sizeof(f64));
    f64 *mat = xcalloc(m * m, sizeof(f64));
    f64 *wrk1 = xcalloc(m * n, sizeof(f64));
    f64 *wrk2 = xcalloc(m * n, sizeof(f64));
    f64 *wrk3 = xcalloc(m * n, sizeof(f64));

    make_indef_diag_d(m, n_pos, mat);

    /* Fill U with random values */
    for (uint64_t i = 0; i < m * n; i++)
        U[i] = (f64)rand() / RAND_MAX - 0.5;

    uint64_t ncols = d_svqb_mat(m, n, 1e-14, 'n', U, mat, wrk1, wrk2, wrk3);
    printf("  Returned %lu columns (expected %lu)\n",
           (unsigned long)ncols, (unsigned long)n);

    f64 err = ortho_error_mat_d(m, n, U, mat);
    printf("  ||U^H*mat*U - I_sig||_F = %.3e (tol = %.3e)\n", err, TOL_F64 * n);

    int pass = (ncols == n) && (err < TOL_F64 * n);
    safe_free((void**)&U); safe_free((void**)&mat); safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3);
    return pass;
}

int test_svqb_mat_z(void) {
    const uint64_t m = 100, n = 10;
    const uint64_t n_pos = 60;

    c64 *U = xcalloc(m * n, sizeof(c64));
    c64 *mat = xcalloc(m * m, sizeof(c64));
    c64 *wrk1 = xcalloc(m * n, sizeof(c64));
    c64 *wrk2 = xcalloc(m * n, sizeof(c64));
    c64 *wrk3 = xcalloc(m * n, sizeof(c64));

    make_indef_diag_z(m, n_pos, mat);

    for (uint64_t i = 0; i < m * n; i++)
        U[i] = ((f64)rand()/RAND_MAX - 0.5) + I*((f64)rand()/RAND_MAX - 0.5);

    uint64_t ncols = z_svqb_mat(m, n, 1e-14, 'n', U, mat, wrk1, wrk2, wrk3);
    printf("  Returned %lu columns (expected %lu)\n",
           (unsigned long)ncols, (unsigned long)n);

    f64 err = ortho_error_mat_z(m, n, U, mat);
    printf("  ||U^H*mat*U - I_sig||_F = %.3e (tol = %.3e)\n", err, TOL_F64 * n);

    int pass = (ncols == n) && (err < TOL_F64 * n);
    safe_free((void**)&U); safe_free((void**)&mat); safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3);
    return pass;
}

/* Also test with positive-definite mat = I to ensure it still works */
int test_svqb_mat_identity_d(void) {
    const uint64_t m = 100, n = 10;

    f64 *U = xcalloc(m * n, sizeof(f64));
    f64 *mat = xcalloc(m * m, sizeof(f64));
    f64 *wrk1 = xcalloc(m * n, sizeof(f64));
    f64 *wrk2 = xcalloc(m * n, sizeof(f64));
    f64 *wrk3 = xcalloc(m * n, sizeof(f64));

    /* mat = I */
    for (uint64_t i = 0; i < m; i++) mat[i + i*m] = 1.0;

    for (uint64_t i = 0; i < m * n; i++)
        U[i] = (f64)rand() / RAND_MAX - 0.5;

    d_svqb_mat(m, n, 1e-14, 'n', U, mat, wrk1, wrk2, wrk3);

    /* Check standard orthonormality: ||U^T*U - I||_F */
    f64 *G = xcalloc(n * n, sizeof(f64));
    d_gemm_tn(n, n, m, 1.0, U, U, 0.0, G);
    for (uint64_t i = 0; i < n; i++) G[i + i*n] -= 1.0;
    f64 err = d_nrm2(n * n, G);
    printf("  ||U^T*U - I||_F = %.3e (tol = %.3e)\n", err, TOL_F64 * n);

    int pass = (err < TOL_F64 * n);
    safe_free((void**)&U); safe_free((void**)&mat); safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3); safe_free((void**)&G);
    return pass;
}

int main(void) {
    srand((unsigned)time(NULL));

    printf("Testing svqb_mat...\n");

    printf("\nTest 1: svqb_mat_d with identity matrix\n");
    int t1 = test_svqb_mat_identity_d();
    printf("  Result: %s\n", t1 ? "PASS" : "FAIL");

    printf("\nTest 2: svqb_mat_d with indefinite diagonal matrix\n");
    int t2 = test_svqb_mat_d();
    printf("  Result: %s\n", t2 ? "PASS" : "FAIL");

    printf("\nTest 3: svqb_mat_z with indefinite diagonal matrix\n");
    int t3 = test_svqb_mat_z();
    printf("  Result: %s\n", t3 ? "PASS" : "FAIL");

    if (t1 && t2 && t3) {
        printf("\nAll tests PASSED\n");
        return 0;
    } else {
        printf("\nSome tests FAILED\n");
        return 1;
    }
}
