/**
 * @file test_blas.c
 * @brief Unit tests for BLAS/LAPACK wrappers
 *
 * Compile: gcc -o test_blas test_blas.c -I../include/lobpcg -lopenblas -lm
 * Run: ./test_blas
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "blas_wrapper.h"

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
#define ASSERT_NEAR_C64(a, b, tol) ASSERT(cabs((a) - (b)) < (tol))

/* ====================================================================
 * BLAS Level 1 Tests
 * ==================================================================== */

TEST(d_nrm2) {
    f64 x[] = {3.0, 4.0};
    f64 result = d_nrm2(2, x);
    ASSERT_NEAR(result, 5.0, TOL_F64);
}

TEST(z_nrm2) {
    c64 x[] = {3.0 + 0.0*I, 0.0 + 4.0*I};
    f64 result = z_nrm2(2, x);
    ASSERT_NEAR(result, 5.0, TOL_F64);
}

TEST(d_axpy) {
    f64 x[] = {1.0, 2.0, 3.0};
    f64 y[] = {4.0, 5.0, 6.0};
    d_axpy(3, 2.0, x, y);
    ASSERT_NEAR(y[0], 6.0, TOL_F64);
    ASSERT_NEAR(y[1], 9.0, TOL_F64);
    ASSERT_NEAR(y[2], 12.0, TOL_F64);
}

TEST(z_axpy) {
    c64 x[] = {1.0 + 1.0*I, 2.0 + 2.0*I};
    c64 y[] = {1.0 + 0.0*I, 0.0 + 1.0*I};
    z_axpy(2, 2.0 + 0.0*I, x, y);
    ASSERT_NEAR_C64(y[0], 3.0 + 2.0*I, TOL_F64);
    ASSERT_NEAR_C64(y[1], 4.0 + 5.0*I, TOL_F64);
}

TEST(d_scal) {
    f64 x[] = {1.0, 2.0, 3.0};
    d_scal(3, 2.0, x);
    ASSERT_NEAR(x[0], 2.0, TOL_F64);
    ASSERT_NEAR(x[1], 4.0, TOL_F64);
    ASSERT_NEAR(x[2], 6.0, TOL_F64);
}

TEST(d_copy) {
    f64 x[] = {1.0, 2.0, 3.0};
    f64 y[] = {0.0, 0.0, 0.0};
    d_copy(3, x, y);
    ASSERT_NEAR(y[0], 1.0, TOL_F64);
    ASSERT_NEAR(y[1], 2.0, TOL_F64);
    ASSERT_NEAR(y[2], 3.0, TOL_F64);
}

TEST(d_dot) {
    f64 x[] = {1.0, 2.0, 3.0};
    f64 y[] = {4.0, 5.0, 6.0};
    f64 result = d_dot(3, x, y);
    ASSERT_NEAR(result, 32.0, TOL_F64);  /* 1*4 + 2*5 + 3*6 = 32 */
}

TEST(z_dotc) {
    c64 x[] = {1.0 + 1.0*I, 2.0 + 2.0*I};
    c64 y[] = {1.0 + 1.0*I, 2.0 + 2.0*I};
    c64 result = z_dotc(2, x, y);
    /* x^H * y = conj(x) * y = (1-i)(1+i) + (2-2i)(2+2i) = 2 + 8 = 10 */
    ASSERT_NEAR_C64(result, 10.0 + 0.0*I, TOL_F64);
}

/* ====================================================================
 * BLAS Level 3 Tests
 * ==================================================================== */

TEST(d_gemm_nn) {
    /* A = [1 2; 3 4], B = [5 6; 7 8], C = A*B = [19 22; 43 50] */
    /* Column-major: A = {1,3,2,4}, B = {5,7,6,8} */
    f64 A[] = {1.0, 3.0, 2.0, 4.0};
    f64 B[] = {5.0, 7.0, 6.0, 8.0};
    f64 C[] = {0.0, 0.0, 0.0, 0.0};
    d_gemm_nn(2, 2, 2, 1.0, A, B, 0.0, C);
    ASSERT_NEAR(C[0], 19.0, TOL_F64);
    ASSERT_NEAR(C[1], 43.0, TOL_F64);
    ASSERT_NEAR(C[2], 22.0, TOL_F64);
    ASSERT_NEAR(C[3], 50.0, TOL_F64);
}

TEST(d_gemm_tn) {
    /* C = A^T * B where A = [1 3; 2 4]^T = [1 2; 3 4] stored, B = [5 6; 7 8] */
    /* Result: [1 2; 3 4] * [5 6; 7 8] = [19 22; 43 50] */
    f64 A[] = {1.0, 2.0, 3.0, 4.0};  /* stored as 2x2, will transpose to get [1 3; 2 4] */
    f64 B[] = {5.0, 7.0, 6.0, 8.0};
    f64 C[] = {0.0, 0.0, 0.0, 0.0};
    d_gemm_tn(2, 2, 2, 1.0, A, B, 0.0, C);
    ASSERT_NEAR(C[0], 19.0, TOL_F64);
    ASSERT_NEAR(C[1], 43.0, TOL_F64);
    ASSERT_NEAR(C[2], 22.0, TOL_F64);
    ASSERT_NEAR(C[3], 50.0, TOL_F64);
}

TEST(z_gemm_nn) {
    /* Simple 2x2 complex multiplication */
    c64 A[] = {1.0+0.0*I, 0.0+0.0*I, 0.0+0.0*I, 1.0+0.0*I};  /* Identity */
    c64 B[] = {1.0+1.0*I, 2.0+2.0*I, 3.0+3.0*I, 4.0+4.0*I};
    c64 C[] = {0.0, 0.0, 0.0, 0.0};
    z_gemm_nn(2, 2, 2, 1.0+0.0*I, A, B, 0.0+0.0*I, C);
    ASSERT_NEAR_C64(C[0], B[0], TOL_F64);
    ASSERT_NEAR_C64(C[1], B[1], TOL_F64);
    ASSERT_NEAR_C64(C[2], B[2], TOL_F64);
    ASSERT_NEAR_C64(C[3], B[3], TOL_F64);
}

TEST(d_syrk) {
    /* C = A^T * A where A = [1 2; 3 4] (2x2, col-major) */
    /* A^T*A = [1 3; 2 4] * [1 2; 3 4] = [10 14; 14 20] */
    f64 A[] = {1.0, 3.0, 2.0, 4.0};  /* column-major */
    f64 C[] = {0.0, 0.0, 0.0, 0.0};
    d_syrk(2, 2, 1.0, A, 0.0, C);
    ASSERT_NEAR(C[0], 10.0, TOL_F64);  /* C[0,0] */
    ASSERT_NEAR(C[2], 14.0, TOL_F64);  /* C[0,1] upper */
    ASSERT_NEAR(C[3], 20.0, TOL_F64);  /* C[1,1] */
}

TEST(z_herk) {
    /* C = A^H * A where A = [1+i, 2+i] (1x2, col-major) */
    /* A^H*A = [1-i; 2-i] * [1+i, 2+i] = [2, 3-i; 3+i, 5] */
    c64 A[] = {1.0+1.0*I, 2.0+1.0*I};
    c64 C[] = {0.0, 0.0, 0.0, 0.0};
    z_herk(1, 2, 1.0, A, 0.0, C);
    ASSERT_NEAR(creal(C[0]), 2.0, TOL_F64);
    ASSERT_NEAR(creal(C[3]), 5.0, TOL_F64);
}

TEST(d_trsm_lln) {
    /* Solve L*X = B where L = [2 0; 1 3], B = [4; 5] */
    /* X = [2; 1] */
    f64 L[] = {2.0, 1.0, 0.0, 3.0};  /* column-major lower triangular */
    f64 B[] = {4.0, 5.0};
    d_trsm_lln(2, 1, 1.0, L, B);
    ASSERT_NEAR(B[0], 2.0, TOL_F64);
    ASSERT_NEAR(B[1], 1.0, TOL_F64);
}

/* ====================================================================
 * LAPACK Tests
 * ==================================================================== */

TEST(d_potrf) {
    /* Cholesky of A = [4 2; 2 5] -> R^H*R, R = [2 1; 0 2] (upper triangle) */
    f64 A[] = {4.0, 2.0, 2.0, 5.0};
    int info = d_potrf(2, A);
    ASSERT(info == 0);
    ASSERT_NEAR(A[0], 2.0, TOL_F64);  /* R[0,0] */
    ASSERT_NEAR(A[2], 1.0, TOL_F64);  /* R[0,1] */
    ASSERT_NEAR(A[3], 2.0, TOL_F64);  /* R[1,1] */
}

TEST(d_syev) {
    /* Eigenvalues of A = [2 1; 1 2] are 1 and 3 */
    f64 A[] = {2.0, 1.0, 1.0, 2.0};
    f64 w[2];
    int info = d_syev(2, A, w);
    ASSERT(info == 0);
    ASSERT_NEAR(w[0], 1.0, TOL_F64);
    ASSERT_NEAR(w[1], 3.0, TOL_F64);
}

TEST(z_heev) {
    /* Eigenvalues of Hermitian A = [2 i; -i 2] are 1 and 3 */
    c64 A[] = {2.0+0.0*I, 0.0-1.0*I, 0.0+1.0*I, 2.0+0.0*I};
    f64 w[2];
    int info = z_heev(2, A, w);
    ASSERT(info == 0);
    ASSERT_NEAR(w[0], 1.0, TOL_F64);
    ASSERT_NEAR(w[1], 3.0, TOL_F64);
}

TEST(d_geev) {
    /* Eigenvalues of A = [0 -1; 1 0] are +i and -i */
    f64 A[] = {0.0, 1.0, -1.0, 0.0};
    f64 wr[2], wi[2];
    f64 VR[4];
    int info = d_geev(2, A, wr, wi, VR);
    ASSERT(info == 0);
    /* Eigenvalues are purely imaginary: 0 +/- 1i */
    ASSERT_NEAR(wr[0], 0.0, TOL_F64);
    ASSERT_NEAR(wr[1], 0.0, TOL_F64);
    ASSERT_NEAR(fabs(wi[0]), 1.0, TOL_F64);
    ASSERT_NEAR(fabs(wi[1]), 1.0, TOL_F64);
}

TEST(d_ggev) {
    /* Generalized eigenvalue: A*x = lambda*B*x */
    /* A = [2 0; 0 4], B = [1 0; 0 2] -> eigenvalues are 2/1=2 and 4/2=2 */
    f64 A[] = {2.0, 0.0, 0.0, 4.0};
    f64 B[] = {1.0, 0.0, 0.0, 2.0};
    f64 alphar[2], alphai[2], beta[2];
    f64 VR[4];
    int info = d_ggev(2, A, B, alphar, alphai, beta, VR);
    ASSERT(info == 0);
    /* Both eigenvalues should be 2 */
    ASSERT_NEAR(alphar[0]/beta[0], 2.0, TOL_F64);
    ASSERT_NEAR(alphar[1]/beta[1], 2.0, TOL_F64);
}

TEST(d_trcon) {
    /* Condition number of identity is 1 */
    f64 L[] = {1.0, 0.0, 0.0, 1.0};
    f64 rcond;
    int info = d_trcon('1', 2, L, &rcond);
    ASSERT(info == 0);
    ASSERT_NEAR(rcond, 1.0, TOL_F64);
}

/* ====================================================================
 * Float precision tests (spot check)
 * ==================================================================== */

TEST(s_nrm2) {
    f32 x[] = {3.0f, 4.0f};
    f32 result = s_nrm2(2, x);
    ASSERT_NEAR(result, 5.0f, TOL_F32);
}

TEST(s_gemm_nn) {
    f32 A[] = {1.0f, 3.0f, 2.0f, 4.0f};
    f32 B[] = {5.0f, 7.0f, 6.0f, 8.0f};
    f32 C[] = {0.0f, 0.0f, 0.0f, 0.0f};
    s_gemm_nn(2, 2, 2, 1.0f, A, B, 0.0f, C);
    ASSERT_NEAR(C[0], 19.0f, TOL_F32);
    ASSERT_NEAR(C[3], 50.0f, TOL_F32);
}

TEST(c_heev) {
    c32 A[] = {2.0f+0.0f*I, 0.0f-1.0f*I, 0.0f+1.0f*I, 2.0f+0.0f*I};
    f32 w[2];
    int info = c_heev(2, A, w);
    ASSERT(info == 0);
    ASSERT_NEAR(w[0], 1.0f, TOL_F32);
    ASSERT_NEAR(w[1], 3.0f, TOL_F32);
}

/* ====================================================================
 * Type-generic macro tests
 * ==================================================================== */

TEST(generic_nrm2) {
    f64 x[] = {3.0, 4.0};
    f64 result = nrm2(2, x);
    ASSERT_NEAR(result, 5.0, TOL_F64);
}

TEST(generic_dot) {
    f64 x[] = {1.0, 2.0, 3.0};
    f64 y[] = {4.0, 5.0, 6.0};
    f64 result = dot(3, x, y);
    ASSERT_NEAR(result, 32.0, TOL_F64);
}

TEST(generic_eig) {
    f64 A[] = {2.0, 1.0, 1.0, 2.0};
    f64 w[2];
    int info = eig(2, A, w);
    ASSERT(info == 0);
    ASSERT_NEAR(w[0], 1.0, TOL_F64);
    ASSERT_NEAR(w[1], 3.0, TOL_F64);
}

/* ====================================================================
 * Main
 * ==================================================================== */

int main(void) {
    printf("BLAS Level 1 tests:\n");
    RUN(d_nrm2);
    RUN(z_nrm2);
    RUN(d_axpy);
    RUN(z_axpy);
    RUN(d_scal);
    RUN(d_copy);
    RUN(d_dot);
    RUN(z_dotc);

    printf("\nBLAS Level 3 tests:\n");
    RUN(d_gemm_nn);
    RUN(d_gemm_tn);
    RUN(z_gemm_nn);
    RUN(d_syrk);
    RUN(z_herk);
    RUN(d_trsm_lln);

    printf("\nLAPACK tests:\n");
    RUN(d_potrf);
    RUN(d_syev);
    RUN(z_heev);
    RUN(d_geev);
    RUN(d_ggev);
    RUN(d_trcon);

    printf("\nFloat precision tests:\n");
    RUN(s_nrm2);
    RUN(s_gemm_nn);
    RUN(c_heev);

    printf("\nType-generic macro tests:\n");
    RUN(generic_nrm2);
    RUN(generic_dot);
    RUN(generic_eig);

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}
