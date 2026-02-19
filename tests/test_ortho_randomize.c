/**
 * @file test_ortho_randomize.c
 * @brief Test ortho_randomize B-orthogonalization against V
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

/* Test ortho_randomize for double precision */
int test_ortho_randomize_d(void) {
    const uint64_t m = 100;
    const uint64_t n_u = 10;
    const uint64_t n_v = 10;
    const f64 eps_ortho = 1e-14;
    const f64 eps_randomize = 1e-14;

    f64 *U = xcalloc(m * n_u, sizeof(f64));
    f64 *V = xcalloc(m * n_v, sizeof(f64));
    /* wrk1: max(m*n_u, m*n_v) for B*U or B*V; also n*n for Gram in svqb */
    /* wrk2: max(m*n_u, n_v*n_u) for ortho coefs + svqb gemm output */
    /* wrk3: n_u*n_u for svqb transform */
    const uint64_t wrk_mn = m * (n_u > n_v ? n_u : n_v);
    f64 *wrk1 = xcalloc(wrk_mn, sizeof(f64));
    f64 *wrk2 = xcalloc(wrk_mn, sizeof(f64));
    f64 *wrk3 = xcalloc(wrk_mn, sizeof(f64));

    /* Fill U and V with random values */
    for (uint64_t i = 0; i < m * n_u; i++) U[i] = (f64)rand() / RAND_MAX;
    for (uint64_t i = 0; i < m * n_v; i++) V[i] = (f64)rand() / RAND_MAX;

    /* B-orthonormalize V first */
    d_svqb(m, n_v, eps_randomize, 'n', V, wrk1, wrk2, wrk3, NULL);

    /* Orthogonalize U against V */
    uint64_t n_ret = d_ortho_randomize(m, n_u, n_v, eps_ortho, eps_randomize,
                                        U, V, wrk1, wrk2, wrk3, NULL);

    printf("  Returned %lu columns (expected %lu)\n", (unsigned long)n_ret, (unsigned long)n_u);

    /* Check ||V^H * U||_F < tol */
    f64 *VtU = xcalloc(n_v * n_u, sizeof(f64));
    d_gemm_tn(n_v, n_u, m, 1.0, V, U, 0.0, VtU);
    f64 VtU_norm = d_nrm2(n_v * n_u, VtU);
    printf("  ||V^H * U||_F = %.3e (should be < %.3e)\n", VtU_norm, TEST_TOLERANCE);

    /* Check ||U^H * U - I||_F < tol */
    f64 *UtU = xcalloc(n_u * n_u, sizeof(f64));
    d_gemm_tn(n_u, n_u, m, 1.0, U, U, 0.0, UtU);
    for (uint64_t i = 0; i < n_u; i++) UtU[i + i*n_u] -= 1.0;
    f64 UtU_norm = d_nrm2(n_u * n_u, UtU);
    printf("  ||U^H * U - I||_F = %.3e (should be < %.3e)\n", UtU_norm, TEST_TOLERANCE);

    int pass = (VtU_norm < TEST_TOLERANCE) && (UtU_norm < TEST_TOLERANCE);

    safe_free((void**)&U); safe_free((void**)&V); safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3); safe_free((void**)&VtU); safe_free((void**)&UtU);
    return pass;
}

/* Test ortho_randomize for complex double precision */
int test_ortho_randomize_z(void) {
    const uint64_t m = 100;
    const uint64_t n_u = 10;
    const uint64_t n_v = 10;
    const f64 eps_ortho = 1e-14;
    const f64 eps_randomize = 1e-14;

    c64 *U = xcalloc(m * n_u, sizeof(c64));
    c64 *V = xcalloc(m * n_v, sizeof(c64));
    const uint64_t wrk_mn = m * (n_u > n_v ? n_u : n_v);
    c64 *wrk1 = xcalloc(wrk_mn, sizeof(c64));
    c64 *wrk2 = xcalloc(wrk_mn, sizeof(c64));
    c64 *wrk3 = xcalloc(wrk_mn, sizeof(c64));

    /* Fill U and V with random complex values */
    for (uint64_t i = 0; i < m * n_u; i++) {
        U[i] = ((f64)rand() / RAND_MAX) + I * ((f64)rand() / RAND_MAX);
    }
    for (uint64_t i = 0; i < m * n_v; i++) {
        V[i] = ((f64)rand() / RAND_MAX) + I * ((f64)rand() / RAND_MAX);
    }

    /* B-orthonormalize V first */
    z_svqb(m, n_v, eps_randomize, 'n', V, wrk1, wrk2, wrk3, NULL);

    /* Orthogonalize U against V */
    uint64_t n_ret = z_ortho_randomize(m, n_u, n_v, eps_ortho, eps_randomize,
                                        U, V, wrk1, wrk2, wrk3, NULL);

    printf("  Returned %lu columns (expected %lu)\n", (unsigned long)n_ret, (unsigned long)n_u);

    /* Check ||V^H * U||_F < tol */
    c64 *VhU = xcalloc(n_v * n_u, sizeof(c64));
    z_gemm_hn(n_v, n_u, m, 1.0, V, U, 0.0, VhU);
    f64 VhU_norm = z_nrm2(n_v * n_u, VhU);
    printf("  ||V^H * U||_F = %.3e (should be < %.3e)\n", VhU_norm, TEST_TOLERANCE);

    /* Check ||U^H * U - I||_F < tol */
    c64 *UhU = xcalloc(n_u * n_u, sizeof(c64));
    z_gemm_hn(n_u, n_u, m, 1.0, U, U, 0.0, UhU);
    for (uint64_t i = 0; i < n_u; i++) UhU[i + i*n_u] -= 1.0;
    f64 UhU_norm = z_nrm2(n_u * n_u, UhU);
    printf("  ||U^H * U - I||_F = %.3e (should be < %.3e)\n", UhU_norm, TEST_TOLERANCE);

    int pass = (VhU_norm < TEST_TOLERANCE) && (UhU_norm < TEST_TOLERANCE);

    safe_free((void**)&U); safe_free((void**)&V); safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3); safe_free((void**)&VhU); safe_free((void**)&UhU);
    return pass;
}

int main(void) {
    printf("Testing ortho_randomize...\n");

    printf("\nTest 1: ortho_randomize_d (double real)\n");
    int test1 = test_ortho_randomize_d();
    printf("  Result: %s\n", test1 ? "PASS" : "FAIL");

    printf("\nTest 2: ortho_randomize_z (complex double)\n");
    int test2 = test_ortho_randomize_z();
    printf("  Result: %s\n", test2 ? "PASS" : "FAIL");

    if (test1 && test2) {
        printf("\nAll tests PASSED\n");
        return 0;
    } else {
        printf("\nSome tests FAILED\n");
        return 1;
    }
}
