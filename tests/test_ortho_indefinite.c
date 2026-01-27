/**
 * @file test_ortho_indefinite.c
 * @brief Unit tests for indefinite B-orthogonalization
 *
 * Tests ortho_indefinite with an indefinite B matrix (diagonal with ±1).
 * Verifies:
 *   1. ||V^H*B*U||_F < tol (U is B-orthogonal to V)
 *   2. ||U^H*B*U - I||_F < tol (U is B-orthonormal)
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
#include "linop.h"

#define TOL_F32 1e-4
#define TOL_F64 1e-10

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) static void test_##name(void)
#define RUN(name) do { \
    printf("  %-50s ", #name); \
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

/* ====================================================================
 * Indefinite diagonal B operator
 *
 * B = diag(1, 1, ..., 1, -1, -1, ..., -1)
 *     (n_pos ones followed by n_neg negative ones)
 * ==================================================================== */

typedef struct {
    uint64_t n;
    uint64_t n_pos;  /* Number of positive entries */
} indef_diag_ctx_t;

static void indef_diag_matvec_d(const LinearOperator_d_t *op, f64 *x, f64 *y) {
    indef_diag_ctx_t *ctx = (indef_diag_ctx_t *)op->ctx->data;
    for (uint64_t i = 0; i < ctx->n; i++) {
        y[i] = (i < ctx->n_pos) ? x[i] : -x[i];
    }
}

static void indef_diag_matvec_z(const LinearOperator_z_t *op, c64 *x, c64 *y) {
    indef_diag_ctx_t *ctx = (indef_diag_ctx_t *)op->ctx->data;
    for (uint64_t i = 0; i < ctx->n; i++) {
        y[i] = (i < ctx->n_pos) ? x[i] : -x[i];
    }
}

static void indef_diag_cleanup(linop_ctx_t *ctx) {
    if (ctx && ctx->data) free(ctx->data);
    if (ctx) free(ctx);
}

static LinearOperator_d_t *create_indef_B_d(uint64_t n, uint64_t n_pos) {
    linop_ctx_t *ctx = malloc(sizeof(linop_ctx_t));
    indef_diag_ctx_t *data = malloc(sizeof(indef_diag_ctx_t));
    data->n = n;
    data->n_pos = n_pos;
    ctx->data = data;
    ctx->data_size = sizeof(indef_diag_ctx_t);
    return linop_create_d(n, n, indef_diag_matvec_d, indef_diag_cleanup, ctx);
}

static LinearOperator_z_t *create_indef_B_z(uint64_t n, uint64_t n_pos) {
    linop_ctx_t *ctx = malloc(sizeof(linop_ctx_t));
    indef_diag_ctx_t *data = malloc(sizeof(indef_diag_ctx_t));
    data->n = n;
    data->n_pos = n_pos;
    ctx->data = data;
    ctx->data_size = sizeof(indef_diag_ctx_t);
    return linop_create_z(n, n, indef_diag_matvec_z, indef_diag_cleanup, ctx);
}

/* ====================================================================
 * Helper functions
 * ==================================================================== */

static void fill_random_d(uint64_t n, f64 *x) {
    for (uint64_t i = 0; i < n; i++)
        x[i] = (f64)rand() / RAND_MAX - 0.5;
}

static void fill_random_z(uint64_t n, c64 *x) {
    for (uint64_t i = 0; i < n; i++)
        x[i] = ((f64)rand()/RAND_MAX - 0.5) + I*((f64)rand()/RAND_MAX - 0.5);
}

/* Fill lower triangle from upper */
static void fill_lower_d(uint64_t n, f64 *A) {
    for (uint64_t j = 0; j < n; j++)
        for (uint64_t i = j + 1; i < n; i++)
            A[i + j*n] = A[j + i*n];
}

static void fill_lower_z(uint64_t n, c64 *A) {
    for (uint64_t j = 0; j < n; j++)
        for (uint64_t i = j + 1; i < n; i++)
            A[i + j*n] = conj(A[j + i*n]);
}

/* Matrix Frobenius norm */
static f64 matrix_nrm_d(uint64_t m, uint64_t n, const f64 *A) {
    f64 sum = 0;
    for (uint64_t i = 0; i < m * n; i++)
        sum += A[i] * A[i];
    return sqrt(sum);
}

static f64 matrix_nrm_z(uint64_t m, uint64_t n, const c64 *A) {
    f64 sum = 0;
    for (uint64_t i = 0; i < m * n; i++)
        sum += cabs(A[i]) * cabs(A[i]);
    return sqrt(sum);
}

/**
 * Compute B-orthogonality error ||V^H*B*U||_F for double precision
 *
 * TODO(human): Implement this verification function.
 *
 * @param m     Number of rows
 * @param n_u   Number of columns in U
 * @param n_v   Number of columns in V
 * @param U     m x n_u matrix
 * @param V     m x n_v matrix
 * @param B     Indefinite B operator
 * @param wrk   Workspace of size m * n_u
 * @return      ||V^H*B*U||_F
 */
static f64 B_ortho_error_d(uint64_t m, uint64_t n_u, uint64_t n_v,
                           const f64 *U, const f64 *V,
                           LinearOperator_d_t *B, f64 *wrk) {
    /* Compute BU = B*U in wrk */
    for (uint64_t j = 0; j < n_u; j++)
        B->matvec(B, (f64*)&U[j * m], &wrk[j * m]);

    /* Compute G = V^T * BU (n_v x n_u) */
    f64 *G = calloc(n_v * n_u, sizeof(f64));
    d_gemm_tn(n_v, n_u, m, 1.0, V, wrk, 0.0, G);

    f64 err = matrix_nrm_d(n_v, n_u, G);
    free(G);
    return err;
}

static f64 B_ortho_error_z(uint64_t m, uint64_t n_u, uint64_t n_v,
                           const c64 *U, const c64 *V,
                           LinearOperator_z_t *B, c64 *wrk) {
    /* Compute BU = B*U in wrk */
    for (uint64_t j = 0; j < n_u; j++)
        B->matvec(B, (c64*)&U[j * m], &wrk[j * m]);

    /* Compute G = V^H * BU (n_v x n_u) */
    c64 *G = calloc(n_v * n_u, sizeof(c64));
    z_gemm_hn(n_v, n_u, m, 1.0, V, wrk, 0.0, G);

    f64 err = matrix_nrm_z(n_v, n_u, G);
    free(G);
    return err;
}

/**
 * Compute B-orthonormality error for INDEFINITE B
 *
 * For indefinite B, U^H*B*U should be a SIGNATURE matrix (diagonal with ±1),
 * not necessarily the identity. We check:
 *   - Off-diagonal elements are near zero
 *   - Diagonal elements are near ±1
 *
 * Returns: max(||off-diag||_F, max_i ||G_ii| - 1|)
 */
static f64 B_orthonorm_error_d(uint64_t m, uint64_t n,
                               const f64 *U, LinearOperator_d_t *B, f64 *wrk) {
    /* BU = B*U */
    for (uint64_t j = 0; j < n; j++)
        B->matvec(B, (f64*)&U[j * m], &wrk[j * m]);

    /* G = U^T * BU */
    f64 *G = calloc(n * n, sizeof(f64));
    d_gemm_tn(n, n, m, 1.0, U, wrk, 0.0, G);

    /* Check off-diagonal elements should be zero */
    f64 off_diag_err = 0;
    for (uint64_t j = 0; j < n; j++) {
        for (uint64_t i = 0; i < n; i++) {
            if (i != j) {
                off_diag_err += G[i + j * n] * G[i + j * n];
            }
        }
    }
    off_diag_err = sqrt(off_diag_err);

    /* Check diagonal elements should be ±1 */
    f64 diag_err = 0;
    for (uint64_t i = 0; i < n; i++) {
        f64 d = fabs(G[i + i * n]) - 1.0;  /* |G_ii| should be 1 */
        if (fabs(d) > diag_err) diag_err = fabs(d);
    }

    free(G);
    return (off_diag_err > diag_err) ? off_diag_err : diag_err;
}

static f64 B_orthonorm_error_z(uint64_t m, uint64_t n,
                               const c64 *U, LinearOperator_z_t *B, c64 *wrk) {
    /* BU = B*U */
    for (uint64_t j = 0; j < n; j++)
        B->matvec(B, (c64*)&U[j * m], &wrk[j * m]);

    /* G = U^H * BU */
    c64 *G = calloc(n * n, sizeof(c64));
    z_gemm_hn(n, n, m, 1.0, U, wrk, 0.0, G);

    /* Check off-diagonal elements should be zero */
    f64 off_diag_err = 0;
    for (uint64_t j = 0; j < n; j++) {
        for (uint64_t i = 0; i < n; i++) {
            if (i != j) {
                off_diag_err += cabs(G[i + j * n]) * cabs(G[i + j * n]);
            }
        }
    }
    off_diag_err = sqrt(off_diag_err);

    /* Check diagonal elements should be ±1 (real for Hermitian) */
    f64 diag_err = 0;
    for (uint64_t i = 0; i < n; i++) {
        f64 d = fabs(creal(G[i + i * n])) - 1.0;  /* |Re(G_ii)| should be 1 */
        f64 im = fabs(cimag(G[i + i * n]));       /* Im(G_ii) should be 0 */
        f64 err = (fabs(d) > im) ? fabs(d) : im;
        if (err > diag_err) diag_err = err;
    }

    free(G);
    return (off_diag_err > diag_err) ? off_diag_err : diag_err;
}

/* ====================================================================
 * Double precision tests
 * ==================================================================== */

TEST(d_ortho_indefinite_basic) {
    /*
     * Test basic indefinite orthogonalization:
     * - Create indefinite B = diag(1,1,1,...,-1,-1,-1)
     * - Generate random V, B-orthonormalize it
     * - Generate random U
     * - Call ortho_indefinite
     * - Verify U is B-orthogonal to V and B-orthonormal
     */
    const uint64_t m = 100;    /* Problem size */
    const uint64_t n_v = 5;    /* External basis size */
    const uint64_t n_u = 8;    /* Candidate basis size */
    const uint64_t n_pos = 60; /* Positive entries in B */

    /* Create indefinite B */
    LinearOperator_d_t *B = create_indef_B_d(m, n_pos);

    /* Allocate matrices */
    f64 *V = calloc(m * n_v, sizeof(f64));
    f64 *U = calloc(m * n_u, sizeof(f64));
    uint64_t wrk_size = m * (n_u > n_v ? n_u : n_v);
    uint64_t coef_size = (n_v > n_u ? n_v : n_u) * (n_v > n_u ? n_v : n_u);
    f64 *wrk1 = calloc(wrk_size, sizeof(f64));
    f64 *wrk2 = calloc(wrk_size, sizeof(f64));
    f64 *wrk3 = calloc(coef_size, sizeof(f64));

    /* Fill V and U with random values */
    fill_random_d(m * n_v, V);
    fill_random_d(m * n_u, U);

    /* B-orthonormalize V using SVQB */
    d_svqb(m, n_v, 1e-14, 'n', V, wrk1, wrk2, wrk3, B);

    /* Compute signature matrix for V */
    f64 *sig = calloc(n_v * n_v, sizeof(f64));
    for (uint64_t j = 0; j < n_v; j++)
        B->matvec(B, &V[j * m], &wrk1[j * m]);
    d_gemm_tn(n_v, n_v, m, 1.0, V, wrk1, 0.0, sig);
    fill_lower_d(n_v, sig);

    /* Call ortho_indefinite */
    d_ortho_indefinite(m, n_u, n_v, TOL_F64, 1e-14, U, V, sig, wrk1, wrk2, wrk3, B);

    /* Verify ||V^H*B*U||_F < tol */
    f64 ortho_err = B_ortho_error_d(m, n_u, n_v, U, V, B, wrk1);
    printf("ortho_err=%.2e ", ortho_err);
    ASSERT(ortho_err < TOL_F64 * n_u);

    /* Verify ||U^H*B*U - I||_F < tol */
    f64 norm_err = B_orthonorm_error_d(m, n_u, U, B, wrk1);
    printf("norm_err=%.2e ", norm_err);
    ASSERT(norm_err < TOL_F64 * n_u);

    /* Cleanup */
    free(V); free(U); free(sig);
    free(wrk1); free(wrk2); free(wrk3);
    linop_destroy_d(B);
}

TEST(d_ortho_indefinite_no_sig) {
    /*
     * Test with sig=NULL (signature computed internally)
     */
    const uint64_t m = 80;
    const uint64_t n_v = 4;
    const uint64_t n_u = 6;
    const uint64_t n_pos = 50;

    LinearOperator_d_t *B = create_indef_B_d(m, n_pos);

    f64 *V = calloc(m * n_v, sizeof(f64));
    f64 *U = calloc(m * n_u, sizeof(f64));
    uint64_t wrk_size = m * (n_u > n_v ? n_u : n_v);
    uint64_t coef_size = (n_v > n_u ? n_v : n_u) * (n_v > n_u ? n_v : n_u);
    f64 *wrk1 = calloc(wrk_size, sizeof(f64));
    f64 *wrk2 = calloc(wrk_size, sizeof(f64));
    f64 *wrk3 = calloc(coef_size, sizeof(f64));

    fill_random_d(m * n_v, V);
    fill_random_d(m * n_u, U);

    /* B-orthonormalize V */
    d_svqb(m, n_v, 1e-14, 'n', V, wrk1, wrk2, wrk3, B);

    /* Call ortho_indefinite with sig=NULL */
    d_ortho_indefinite(m, n_u, n_v, TOL_F64, 1e-14, U, V, NULL, wrk1, wrk2, wrk3, B);

    f64 ortho_err = B_ortho_error_d(m, n_u, n_v, U, V, B, wrk1);
    f64 norm_err = B_orthonorm_error_d(m, n_u, U, B, wrk1);
    printf("ortho=%.2e norm=%.2e ", ortho_err, norm_err);
    ASSERT(ortho_err < TOL_F64 * n_u);
    ASSERT(norm_err < TOL_F64 * n_u);

    free(V); free(U);
    free(wrk1); free(wrk2); free(wrk3);
    linop_destroy_d(B);
}

/* ====================================================================
 * Double complex tests
 * ==================================================================== */

TEST(z_ortho_indefinite_basic) {
    const uint64_t m = 100;
    const uint64_t n_v = 5;
    const uint64_t n_u = 8;
    const uint64_t n_pos = 60;

    LinearOperator_z_t *B = create_indef_B_z(m, n_pos);

    c64 *V = calloc(m * n_v, sizeof(c64));
    c64 *U = calloc(m * n_u, sizeof(c64));
    uint64_t wrk_size = m * (n_u > n_v ? n_u : n_v);
    uint64_t coef_size = (n_v > n_u ? n_v : n_u) * (n_v > n_u ? n_v : n_u);
    c64 *wrk1 = calloc(wrk_size, sizeof(c64));
    c64 *wrk2 = calloc(wrk_size, sizeof(c64));
    c64 *wrk3 = calloc(coef_size, sizeof(c64));

    fill_random_z(m * n_v, V);
    fill_random_z(m * n_u, U);

    /* B-orthonormalize V */
    z_svqb(m, n_v, 1e-14, 'n', V, wrk1, wrk2, wrk3, B);

    /* Call ortho_indefinite with sig=NULL */
    z_ortho_indefinite(m, n_u, n_v, TOL_F64, 1e-14, U, V, NULL, wrk1, wrk2, wrk3, B);

    f64 ortho_err = B_ortho_error_z(m, n_u, n_v, U, V, B, wrk1);
    f64 norm_err = B_orthonorm_error_z(m, n_u, U, B, wrk1);
    printf("ortho=%.2e norm=%.2e ", ortho_err, norm_err);
    ASSERT(ortho_err < TOL_F64 * n_u);
    ASSERT(norm_err < TOL_F64 * n_u);

    free(V); free(U);
    free(wrk1); free(wrk2); free(wrk3);
    linop_destroy_z(B);
}

TEST(z_ortho_indefinite_larger) {
    const uint64_t m = 500;
    const uint64_t n_v = 10;
    const uint64_t n_u = 15;
    const uint64_t n_pos = 300;

    LinearOperator_z_t *B = create_indef_B_z(m, n_pos);

    c64 *V = calloc(m * n_v, sizeof(c64));
    c64 *U = calloc(m * n_u, sizeof(c64));
    uint64_t wrk_size = m * (n_u > n_v ? n_u : n_v);
    uint64_t coef_size = (n_v > n_u ? n_v : n_u) * (n_v > n_u ? n_v : n_u);
    c64 *wrk1 = calloc(wrk_size, sizeof(c64));
    c64 *wrk2 = calloc(wrk_size, sizeof(c64));
    c64 *wrk3 = calloc(coef_size, sizeof(c64));

    fill_random_z(m * n_v, V);
    fill_random_z(m * n_u, U);

    z_svqb(m, n_v, 1e-14, 'n', V, wrk1, wrk2, wrk3, B);
    z_ortho_indefinite(m, n_u, n_v, TOL_F64, 1e-14, U, V, NULL, wrk1, wrk2, wrk3, B);

    f64 ortho_err = B_ortho_error_z(m, n_u, n_v, U, V, B, wrk1);
    f64 norm_err = B_orthonorm_error_z(m, n_u, U, B, wrk1);
    printf("ortho=%.2e norm=%.2e ", ortho_err, norm_err);
    ASSERT(ortho_err < TOL_F64 * n_u);
    ASSERT(norm_err < TOL_F64 * n_u);

    free(V); free(U);
    free(wrk1); free(wrk2); free(wrk3);
    linop_destroy_z(B);
}

/* ====================================================================
 * Main
 * ==================================================================== */

int main(void) {
    srand((unsigned)time(NULL));

    printf("Indefinite orthogonalization (double) tests:\n");
    RUN(d_ortho_indefinite_basic);
    RUN(d_ortho_indefinite_no_sig);

    printf("\nIndefinite orthogonalization (double complex) tests:\n");
    RUN(z_ortho_indefinite_basic);
    RUN(z_ortho_indefinite_larger);

    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n");

    return tests_failed > 0 ? 1 : 0;
}
