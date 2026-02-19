/**
 * @file test_svqb.c
 * @brief Unit tests for SVQB orthogonalization
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

#define TOL_F32 1e-5
#define TOL_F64 1e-12

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) static void test_##name(void)
#define RUN(name) do { \
    printf("  %-40s ", #name); \
    test_##name(); \
    printf("[PASS]\n"); \
    tests_passed++; \
} while(0)

#define ASSERT(cond) do { \
    if (!(cond)) { \
        printf("[FAIL] line %d: %s\n", __LINE__, #cond); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, tol) ASSERT(fabs((a) - (b)) < (tol))

/* Fill matrix with random values */
static void fill_random_d(uint64_t n, f64 *x) {
    for (uint64_t i = 0; i < n; i++)
        x[i] = (f64)rand() / RAND_MAX - 0.5;
}

static void fill_random_z(uint64_t n, c64 *x) {
    for (uint64_t i = 0; i < n; i++)
        x[i] = ((f64)rand()/RAND_MAX - 0.5) + I*((f64)rand()/RAND_MAX - 0.5);
}

/* Compute ||U^H*U - I||_F for double */
static f64 ortho_error_d(uint64_t m, uint64_t n, const f64 *U) {
    f64 *G = xcalloc(n * n, sizeof(f64));
    d_syrk(m, n, 1.0, U, 0.0, G);

    /*    for (uint64_t j = 0; j < n; j++) {
      for (uint64_t i = 0; i < n; i++) {
	printf("%.3e\t",G[i+j*n]);
      }
      printf("\n");
      }*/
    
    /* Fill lower and subtract I */
    for (uint64_t j = 0; j < n; j++) {
        for (uint64_t i = j; i < n; i++) {
            if (i == j) G[i + j*n] -= 1.0;
            else G[i + j*n] = G[j + i*n];
        }
    }

    /*    for (uint64_t j = 0; j < n; j++) {
      for (uint64_t i = 0; i < n; i++) {
	printf("%.3e\t",G[i+j*n]);
      }
      printf("\n");
      }*/

    f64 err = d_nrm2(n*n, G);
    safe_free((void**)&G);
    return err;
}

/* Compute ||U^H*U - I||_F for double complex */
static f64 ortho_error_z(uint64_t m, uint64_t n, const c64 *U) {
    c64 *G = xcalloc(n * n, sizeof(c64));
    z_herk(m, n, 1.0, U, 0.0, G);
    /* Fill lower and subtract I */
    for (uint64_t j = 0; j < n; j++) {
        for (uint64_t i = j; i < n; i++) {
            if (i == j) G[i + j*n] -= 1.0;
            else G[i + j*n] = conj(G[j + i*n]);
        }
    }
    f64 err = z_nrm2(n*n, G);
    for (uint64_t j = 0; j < n; j++) {
      for (uint64_t i = 0; i < n; i++) {
	printf("%.3e %.3e\t", creal(G[i+j*n]), cimag(G[i+j*n]));
      }
      printf("\n");
    }
    safe_free((void**)&G);
    return err;
}

/* ====================================================================
 * Double precision tests
 * ==================================================================== */

TEST(d_svqb_identity) {
    /* Random U → svqb → verify ||U^H*U - I||_F < tol */
    const uint64_t m = 100, n = 10;
    f64 *U = xcalloc(m * n, sizeof(f64));
    f64 *wrk1 = xcalloc(n * n, sizeof(f64));
    f64 *wrk2 = xcalloc(m * n, sizeof(f64));
    f64 *wrk3 = xcalloc(m * n, sizeof(f64));

    fill_random_d(m * n, U);

    uint64_t ncols = d_svqb(m, n, 1e-14, 'n', U, wrk1, wrk2, wrk3, NULL);
    ASSERT(ncols == n);

    f64 err = ortho_error_d(m, n, U);
    printf("err = %.2e, tol = %.2e ", err, TOL_F64 * n);
    ASSERT(err < TOL_F64 * n);

    safe_free((void**)&U); safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3);
}

TEST(d_svqb_larger) {
    /* Larger test: 1000 x 20 */
    const uint64_t m = 1000, n = 20;
    f64 *U = xcalloc(m * n, sizeof(f64));
    f64 *wrk1 = xcalloc(n * n, sizeof(f64));
    f64 *wrk2 = xcalloc(m * n, sizeof(f64));
    f64 *wrk3 = xcalloc(m * n, sizeof(f64));

    fill_random_d(m * n, U);

    d_svqb(m, n, 1e-14, 'n', U, wrk1, wrk2, wrk3, NULL);

    f64 err = ortho_error_d(m, n, U);
    ASSERT(err < TOL_F64 * n);

    safe_free((void**)&U); safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3);
}

/* ====================================================================
 * Double complex tests
 * ==================================================================== */

TEST(z_svqb_identity) {
    /* Random U → svqb → verify ||U^H*U - I||_F < tol */
    const uint64_t m = 100, n = 10;
    c64 *U = xcalloc(m * n, sizeof(c64));
    c64 *wrk1 = xcalloc(n * n, sizeof(c64));
    c64 *wrk2 = xcalloc(m * n, sizeof(c64));
    c64 *wrk3 = xcalloc(m * n, sizeof(c64));

    fill_random_z(m * n, U);

    uint64_t ncols = z_svqb(m, n, 1e-14, 'n', U, wrk1, wrk2, wrk3, NULL);
    ASSERT(ncols == n);

    f64 err = ortho_error_z(m, n, U);
    ASSERT(err < TOL_F64 * n);

    safe_free((void**)&U); safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3);
}

TEST(z_svqb_larger) {
    /* Larger test: 1000 x 20 */
    const uint64_t m = 1000, n = 20;
    c64 *U = xcalloc(m * n, sizeof(c64));
    c64 *wrk1 = xcalloc(n * n, sizeof(c64));
    c64 *wrk2 = xcalloc(m * n, sizeof(c64));
    c64 *wrk3 = xcalloc(m * n, sizeof(c64));

    fill_random_z(m * n, U);

    z_svqb(m, n, 1e-14, 'n', U, wrk1, wrk2, wrk3, NULL);

    f64 err = ortho_error_z(m, n, U);
    ASSERT(err < TOL_F64 * n);

    safe_free((void**)&U); safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3);
}

/* ====================================================================
 * Single precision tests
 * ==================================================================== */

TEST(s_svqb_identity) {
    const uint64_t m = 100, n = 10;
    f32 *U = xcalloc(m * n, sizeof(f32));
    f32 *wrk1 = xcalloc(n * n, sizeof(f32));
    f32 *wrk2 = xcalloc(m * n, sizeof(f32));
    f32 *wrk3 = xcalloc(m * n, sizeof(f32));

    for (uint64_t i = 0; i < m*n; i++)
        U[i] = (f32)rand() / RAND_MAX - 0.5f;

    s_svqb(m, n, 1e-6f, 'n', U, wrk1, wrk2, wrk3, NULL);

    /* Compute error */
    f32 *G = xcalloc(n * n, sizeof(f32));
    s_syrk(m, n, 1.0f, U, 0.0f, G);
    for (uint64_t i = 0; i < n; i++) G[i + i*n] -= 1.0f;
    f32 err = s_nrm2(n*n, G);
    ASSERT(err < TOL_F32 * n);

    safe_free((void**)&U); safe_free((void**)&wrk1); safe_free((void**)&wrk2); safe_free((void**)&wrk3); safe_free((void**)&G);
}

/* ====================================================================
 * Main
 * ==================================================================== */

int main(void) {
    srand((unsigned)time(NULL));

    printf("SVQB double precision tests:\n");
    RUN(d_svqb_identity);
    RUN(d_svqb_larger);

    printf("\nSVQB double complex tests:\n");
    RUN(z_svqb_identity);
    RUN(z_svqb_larger);

    printf("\nSVQB single precision tests:\n");
    RUN(s_svqb_identity);

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}
